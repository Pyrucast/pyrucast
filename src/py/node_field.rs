//! Python wrappers for [`crate::containers::node_field::SubNodeField`] and
//! [`crate::containers::node_field::NodeField`].

use crate::aggregate::Aggregate;
use crate::atoms::Band;
use crate::atoms::NodeId;
use crate::containers::field::SubField;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::handle::Handle;
use crate::py::mesh::{PyMesh, PySubMesh};
use crate::py::node::PyNode;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

/// Resolve a node list from the several shapes a caller may pass to a
/// batch read: a list of `Node`, a POI1 `SubMesh`, or a `Mesh` (its
/// submeshes' nodes taken in connectivity order — one entry per point for
/// a POI1 mesh). Order is preserved; duplicates are kept.
fn extract_node_ids(obj: &Bound<'_, PyAny>) -> PyResult<Vec<NodeId>> {
    if let Ok(sm) = obj.extract::<PyRef<PySubMesh>>() {
        return Ok(sm.handle.read().connectivity().to_vec());
    }
    if let Ok(mesh) = obj.extract::<PyRef<PyMesh>>() {
        let mut ids: Vec<NodeId> = Vec::new();
        for h in &mesh.inner {
            ids.extend_from_slice(h.read().connectivity());
        }
        return Ok(ids);
    }
    if let Ok(nodes) = obj.extract::<Vec<PyRef<'_, PyNode>>>() {
        return Ok(nodes.iter().map(|n| n.as_node().id()).collect());
    }
    Err(PyTypeError::new_err(
        "expected a list of Node, a SubMesh, or a Mesh",
    ))
}

// ─── SubNodeField (view) ────────────────────────────────────────────────────

/// A **view** into one zone of a `NodeField`, obtained by indexing
/// (`node_field[i]`) — never constructed directly. Build at the parent
/// level instead: `NodeField(support, components)`, composed with `|`.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "SubNodeField")]
pub struct PySubNodeField {
    pub(crate) handle: Handle<SubNodeField>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PySubNodeField {
    /// Number of nodes in the support.
    fn node_count(&self) -> PyResult<usize> {
        Ok(self.handle.read().node_count())
    }

    /// Number of components stored per node.
    fn component_count(&self) -> PyResult<usize> {
        Ok(self.handle.read().component_count())
    }

    /// Component names, in order.
    fn components(&self) -> PyResult<Vec<String>> {
        Ok(self.handle.read().components().to_vec())
    }

    /// Value at node index `node_idx`, component index `comp_idx`.
    fn get(&self, node_idx: usize, comp_idx: usize) -> PyResult<f64> {
        Ok(self.handle.read().get(node_idx, comp_idx)?)
    }

    /// Set the value at node index `node_idx`, component index `comp_idx`.
    fn set(&self, node_idx: usize, comp_idx: usize, value: f64) -> PyResult<()> {
        self.handle.write().set(node_idx, comp_idx, value)?;
        Ok(())
    }

    /// Value at `node`, component index `comp_idx`.
    fn get_by_node(&self, node: PyRef<'_, PyNode>, comp_idx: usize) -> PyResult<f64> {
        let nid = node.as_node().id();
        Ok(self.handle.read().get_by_node(nid, comp_idx)?)
    }

    /// Set the value at `node`, component index `comp_idx`.
    fn set_by_node(&self, node: PyRef<'_, PyNode>, comp_idx: usize, value: f64) -> PyResult<()> {
        let nid = node.as_node().id();
        self.handle.write().set_by_node(nid, comp_idx, value)?;
        Ok(())
    }

    /// Index of component `name`, or `None` if unknown.
    fn component_index(&self, name: &str) -> PyResult<Option<usize>> {
        Ok(self.handle.read().component_index(name))
    }

    /// All component values at node index `node_idx`, in order.
    fn node_values(&self, node_idx: usize) -> PyResult<Vec<f64>> {
        Ok(self.handle.read().node_values(node_idx)?.to_vec())
    }

    /// The POI1 `SubMesh` this sub-field is supported on.
    fn support_submesh(&self) -> PyResult<PySubMesh> {
        let sm = self.handle.read().support_submesh()?;
        Ok(PySubMesh {
            handle: Handle::new(sm),
        })
    }

    /// The POI1 `Mesh` this sub-field is supported on.
    fn support_mesh(&self) -> PyResult<PyMesh> {
        let mesh = self.handle.read().support_mesh()?;
        Ok(PyMesh { inner: mesh })
    }

    /// Value at `node` for the named `component`.
    fn value(&self, node: PyRef<'_, PyNode>, component: &str) -> PyResult<f64> {
        let nid = node.as_node().id();
        Ok(self.handle.read().value(nid, component)?)
    }

    /// Set the value at `node` for the named `component`.
    fn set_value(&self, node: PyRef<'_, PyNode>, component: &str, value: f64) -> PyResult<()> {
        let nid = node.as_node().id();
        self.handle.write().set_value(nid, component, value)?;
        Ok(())
    }

    /// Smallest value of the named `component` — or, called without one, the
    /// smallest value of the **whole** field, every component pooled. Pooling
    /// reads the field as the flat list of its values: on components carrying
    /// different units it answers "the smallest number in there", not a
    /// physical quantity.
    #[pyo3(signature = (component=None))]
    fn min(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::min(&*self.handle.read(), component)?)
    }

    /// Largest value of the named `component` — or, called without one, the
    /// largest value of the **whole** field, every component pooled (see
    /// `min`).
    #[pyo3(signature = (component=None))]
    fn max(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::max(&*self.handle.read(), component)?)
    }

    /// Sum of the named `component` over the support (Σ over nodes) — the
    /// resultant of a nodal force field, one component at a time. Empty sums
    /// to `0.0`.
    fn sum(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::sum(&*self.handle.read(), component)?)
    }

    /// Add `scalar` to every value of `component` (in place).
    fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        self.handle.write().add_to_component(component, scalar)?;
        Ok(())
    }

    /// Subtract `scalar` from every value of `component` (in place).
    fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        self.handle.write().sub_to_component(component, scalar)?;
        Ok(())
    }

    /// Multiply every value of `component` by `scalar` (in place).
    fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        self.handle.write().mul_to_component(component, scalar)?;
        Ok(())
    }

    /// Divide every value of `component` by `scalar` (in place).
    fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        self.handle.write().div_to_component(component, scalar)?;
        Ok(())
    }

    // ── Arithmetic operators (return a new sub-field) ───────────────────
    //
    // `rhs` may be a float (scalar broadcast over every node × component) or
    // another `SubNodeField` (per-component union with passthrough on a shared
    // support). Division does not guard against zero (inf/nan).

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PySubNodeField> {
        self.scalar_or_combine(rhs, |a, b| a + b)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PySubNodeField> {
        self.scalar_or_combine(rhs, |a, b| a - b)
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PySubNodeField> {
        self.scalar_or_combine(rhs, |a, b| a * b)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PySubNodeField> {
        self.scalar_or_combine(rhs, |a, b| a / b)
    }

    /// `field ** exponent` — element-wise power, same dispatch as the other
    /// operators (float exponent → broadcast; `SubNodeField` → strict
    /// element-by-element). The ternary `pow(x, y, z)` modulo form is
    /// rejected (meaningless on floats).
    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: &Bound<'_, PyAny>,
    ) -> PyResult<PySubNodeField> {
        if !modulo.is_none() {
            return Err(PyTypeError::new_err(
                "field ** exponent does not support a modulo argument",
            ));
        }
        self.scalar_or_combine(exponent, |a, b| a.powf(b))
    }

    /// `subfield[node, "UX"] = v` — raises if the node or component is absent.
    fn __setitem__(&self, key: (PyRef<'_, PyNode>, String), value: f64) -> PyResult<()> {
        let (node, comp) = key;
        let nid = node.as_node().id();
        self.handle.write().set_value(nid, &comp, value)?;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", *self.handle.read()))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", *self.handle.read()))
    }
}

impl PySubNodeField {
    /// Dispatch an arithmetic operator: float → scalar broadcast,
    /// `SubNodeField` → `merge_components` (union of components, passthrough).
    fn scalar_or_combine(
        &self,
        rhs: &Bound<'_, PyAny>,
        op: fn(f64, f64) -> f64,
    ) -> PyResult<PySubNodeField> {
        if let Ok(s) = rhs.extract::<f64>() {
            let out = self.handle.read().map_all(|v| op(v, s));
            Ok(PySubNodeField {
                handle: Handle::new(out),
            })
        } else if let Ok(other) = rhs.extract::<PyRef<PySubNodeField>>() {
            let a = (*self.handle.read()).clone();
            let b = (*other.handle.read()).clone();
            Ok(PySubNodeField {
                handle: Handle::new(a.merge_components(&b, op)?),
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand: expected a float or a SubNodeField",
            ))
        }
    }
}

crate::impl_dump_pymethod!(handle PySubNodeField, handle);

// ─── NodeField (aggregate) ──────────────────────────────────────────────────

/// A field of values carried by mesh nodes — one `SubNodeField` block per
/// zone, with possibly different components from one zone to the next.
///
/// Build with `NodeField(support, components)` where `support` is a `Mesh`
/// (one sub-field per submesh) or a single `SubMesh`; index it
/// (`field[i]`) to reach a `SubNodeField`, compose zones with `|`. Reads
/// (`field.value(node, "T")`) take the first zone defining the pair;
/// `field.check()` verifies that zones agree on shared interface nodes.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[pyclass(name = "NodeField")]
pub struct PyNodeField {
    pub(crate) inner: NodeField,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl PyNodeField {
    /// `NodeField(support, components)` — zero-initialized field. With a
    /// `Mesh` support: one `SubNodeField` per submesh, each on the
    /// distinct nodes of its zone. With a `SubMesh`: a single zone.
    #[new]
    fn py_new(support: &Bound<'_, PyAny>, components: Vec<String>) -> PyResult<Self> {
        let inner = if let Ok(mesh) = support.extract::<PyRef<PyMesh>>() {
            NodeField::new(&mesh.inner, components)?
        } else if let Ok(sm) = support.extract::<PyRef<PySubMesh>>() {
            NodeField::from_submesh(&sm.handle, components)?
        } else {
            return Err(PyTypeError::new_err("expected a Mesh or a SubMesh"));
        };
        Ok(Self { inner })
    }

    /// Explicit `components` list per submesh of `mesh`.
    #[classmethod]
    fn with_components_per_submesh(
        _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
        mesh: PyRef<PyMesh>,
        components_per_submesh: Vec<Vec<String>>,
    ) -> PyResult<Self> {
        let inner = NodeField::with(&mesh.inner, &components_per_submesh)?;
        Ok(Self { inner })
    }

    /// Number of distinct nodes across the zones.
    fn node_count(&self) -> PyResult<usize> {
        Ok(self.inner.node_count()?)
    }

    /// Union of the zones' component names, first-seen order.
    fn components(&self) -> PyResult<Vec<String>> {
        use crate::containers::field::Field;
        Ok(Field::components(&self.inner)?)
    }

    /// Value at `node` for the named `component` — the first zone
    /// defining both wins. Raises if none does.
    fn value(&self, node: PyRef<'_, PyNode>, component: &str) -> PyResult<f64> {
        let nid = node.as_node().id();
        Ok(self.inner.value(nid, component)?)
    }

    /// Values for the named `component` at several nodes, returned in the
    /// **same order** as given — the batch form of `value`. `nodes` is
    /// either a list of `Node`, a POI1 `SubMesh`, or a POI1 `Mesh` (its
    /// nodes taken in connectivity order). The first zone defining each
    /// `(node, component)` pair wins; raises on the first node the field
    /// does not define.
    fn values(&self, nodes: &Bound<'_, PyAny>, component: &str) -> PyResult<Vec<f64>> {
        let ids = extract_node_ids(nodes)?;
        Ok(self.inner.values_at(&ids, component)?)
    }

    /// Verify zone coherence: every `(node, component)` stored by several
    /// zones must hold the same value everywhere. Raises on the first
    /// conflict.
    fn check(&self) -> PyResult<()> {
        Ok(self.inner.check()?)
    }

    /// Smallest value of `component` across the zones defining it — or, called
    /// without a component, the smallest value of the **whole** field, every
    /// component of every zone pooled (see `SubNodeField.min`).
    #[pyo3(signature = (component=None))]
    fn min(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::min(&self.inner, component)?)
    }

    /// Largest value of `component` across the zones defining it — or, called
    /// without a component, the largest value of the **whole** field (see
    /// `min`).
    #[pyo3(signature = (component=None))]
    fn max(&self, component: Option<&str>) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::max(&self.inner, component)?)
    }

    /// Sum of `component` across the zones defining it (Σ over the whole field)
    /// — the resultant of a nodal force field, one component at a time.
    fn sum(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::sum(&self.inner, component)?)
    }

    /// Visualize this field alone, as a **coloured point cloud** over
    /// its support nodes — the POI1 support has no connectivity, so no
    /// surface can be drawn here; use `mesh.plot(field=...)` with the
    /// original mesh for surfaces.
    ///
    /// Same `view` / `save` / `show_axes` / `component` / `vmin` /
    /// `vmax` / `cmap` / `title` semantics as `Mesh.plot`.
    /// `revolve` / `revolve_angle` sweep an axisymmetric plot into its body
    /// of revolution — see `SubMesh.plot`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, component=None, vmin=None, vmax=None, cmap=None, revolve=false, revolve_angle=360.0, title=None))]
    #[allow(clippy::too_many_arguments)]
    fn plot(
        &self,
        view: Option<(f64, f64, f64)>,
        save: Option<std::path::PathBuf>,
        show_axes: bool,
        component: Option<String>,
        vmin: Option<f64>,
        vmax: Option<f64>,
        cmap: Option<String>,
        revolve: bool,
        revolve_angle: f64,
        title: Option<String>,
    ) -> PyResult<()> {
        let view = crate::py::build_view(view, show_axes, revolve, revolve_angle)?;
        let scale = crate::viz::ColorScale {
            cmap: crate::py::mesh::parse_cmap(cmap)?,
            vmin,
            vmax,
        };
        self.inner.plot(
            Some(view),
            save.as_deref(),
            component.as_deref(),
            scale,
            title.as_deref(),
        )?;
        Ok(())
    }

    /// A `Mesh` mirroring this field's supports — the zones' POI1
    /// support submeshes, shared (not copied).
    fn support_mesh(&self) -> PyResult<PyMesh> {
        let mut mesh = crate::containers::mesh::Mesh::empty();
        for h in &self.inner {
            let sm = h.read().support();
            mesh.add_sub(sm)?;
        }
        Ok(PyMesh { inner: mesh })
    }

    // ── Per-component scalar ops (in place, on every zone defining it) ──

    /// Add `scalar` to `component` on every zone that defines it.
    fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        use crate::containers::field::Field;
        self.inner.add_to_component(component, scalar)?;
        Ok(())
    }

    /// Subtract `scalar` from `component` on every zone that defines it.
    fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        use crate::containers::field::Field;
        self.inner.sub_to_component(component, scalar)?;
        Ok(())
    }

    /// Multiply `component` by `scalar` on every zone that defines it.
    fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        use crate::containers::field::Field;
        self.inner.mul_to_component(component, scalar)?;
        Ok(())
    }

    /// Divide `component` by `scalar` on every zone that defines it.
    fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        use crate::containers::field::Field;
        self.inner.div_to_component(component, scalar)?;
        Ok(())
    }

    // ── Arithmetic operators (return a new field) ───────────────────────
    //
    // `rhs`: a float (scalar over every zone), a `NodeField` (same
    // decomposition, strict), or a `SubNodeField` (targeted zone update).

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyNodeField> {
        self.binary(rhs, |a, b| a + b)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyNodeField> {
        self.binary(rhs, |a, b| a - b)
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyNodeField> {
        self.binary(rhs, |a, b| a * b)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<PyNodeField> {
        self.binary(rhs, |a, b| a / b)
    }

    /// `field ** exponent` — element-wise power, same dispatch as the other
    /// operators (float → scalar, `NodeField` → strict same-decomposition,
    /// `SubNodeField` → targeted zone). The ternary `pow(x, y, z)` modulo
    /// form is rejected.
    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: &Bound<'_, PyAny>,
    ) -> PyResult<PyNodeField> {
        if !modulo.is_none() {
            return Err(PyTypeError::new_err(
                "field ** exponent does not support a modulo argument",
            ));
        }
        self.binary(exponent, |a, b| a.powf(b))
    }
}

impl PyNodeField {
    /// Dispatch an arithmetic operator: float → scalar, `NodeField` →
    /// `merge_field` (per `(support, component)`, union/passthrough),
    /// `SubNodeField` → `merge_subfield` (targeted zone update).
    fn binary(&self, rhs: &Bound<'_, PyAny>, op: fn(f64, f64) -> f64) -> PyResult<PyNodeField> {
        use crate::containers::field::Field;
        if let Ok(s) = rhs.extract::<f64>() {
            Ok(PyNodeField {
                inner: self.inner.combine_scalar(op, s)?,
            })
        } else if let Ok(other) = rhs.extract::<PyRef<PyNodeField>>() {
            Ok(PyNodeField {
                inner: self.inner.merge_field(&other.inner, op)?,
            })
        } else if let Ok(sub) = rhs.extract::<PyRef<PySubNodeField>>() {
            let s = (*sub.handle.read()).clone();
            Ok(PyNodeField {
                inner: self.inner.merge_subfield(&s, op)?,
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand: expected a float, a NodeField, or a SubNodeField",
            ))
        }
    }
}

// Polymorphic subscript — **closed block**, undecorated on purpose (see
// `impl_aggregate_pymethods!`): its `.pyi` entries are the hand-written
// overloads submitted just below.
#[pymethods]
impl PySubNodeField {
    /// Indexing dispatches on the key:
    /// - `subfield[node, "UX"]` → the **value** (raises if node/component absent);
    /// - `subfield["UX"]` or `subfield[["UX", "UY"]]` → a **new sub-field** with
    ///   only those components (`filter_components`), so `u1[u2.components()]`
    ///   reprojects `u1` onto `u2`'s component set.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // (node, component) → scalar value.
        if let Ok((node, comp)) = key.extract::<(PyRef<'_, PyNode>, String)>() {
            let nid = node.as_node().id();
            let v = self.handle.read().value(nid, &comp)?;
            return Ok(v.into_pyobject(py)?.into_any().unbind());
        }
        // "comp" or ["comp", …] → component selection (a new sub-field).
        let names = crate::py::ops::field::extract_names(key)?;
        let out =
            crate::containers::field::SubField::select_components(&*self.handle.read(), names)?;
        Ok(Py::new(
            py,
            PySubNodeField {
                handle: Handle::new(out),
            },
        )?
        .into_any())
    }

    /// Comparison sugar → a per-component 0/1 mask (see `mask`). `subfield >= x`
    /// / `> x` / `<= x` / `< x` test every component against the scalar `x`;
    /// `==` / `!=` and non-scalar right-hands fall back to `NotImplemented`.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Ok(x) = other.extract::<f64>() else {
            return Ok(py.NotImplemented());
        };
        let band = match op {
            CompareOp::Ge => Band::new(Some(x), None, None, None),
            CompareOp::Gt => Band::new(None, Some(x), None, None),
            CompareOp::Le => Band::new(None, None, Some(x), None),
            CompareOp::Lt => Band::new(None, None, None, Some(x)),
            CompareOp::Eq | CompareOp::Ne => return Ok(py.NotImplemented()),
        }?;
        let out = crate::ops::node_field::mask_sub(&self.handle.read(), &band, None);
        Ok(Py::new(
            py,
            PySubNodeField {
                handle: Handle::new(out),
            },
        )?
        .into_any())
    }
}

// `__richcmp__` is a pyo3-only spelling: CPython exposes the comparison slots
// as `__ge__`/`__gt__`/`__le__`/`__lt__`, which is what the stub must declare.
#[pymethods]
impl PyNodeField {
    /// Comparison sugar → a per-component 0/1 mask (see `mask`). `field >= x`
    /// / `> x` / `<= x` / `< x` test every component against the scalar `x`;
    /// `==` / `!=` and non-scalar right-hands fall back to `NotImplemented`.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let Ok(x) = other.extract::<f64>() else {
            return Ok(py.NotImplemented());
        };
        let band = match op {
            CompareOp::Ge => Band::new(Some(x), None, None, None),
            CompareOp::Gt => Band::new(None, Some(x), None, None),
            CompareOp::Le => Band::new(None, None, Some(x), None),
            CompareOp::Lt => Band::new(None, None, None, Some(x)),
            CompareOp::Eq | CompareOp::Ne => return Ok(py.NotImplemented()),
        }?;
        let out = crate::ops::node_field::mask(&self.inner, &band, None)?;
        Ok(Py::new(py, PyNodeField { inner: out })?.into_any())
    }
}

#[cfg(feature = "stub-gen")]
pyo3_stub_gen::inventory::submit! {
    pyo3_stub_gen::derive::gen_methods_from_python! { r#"
class PyNodeField:
    def __ge__(self, other: float) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`field >= x` → a fresh `NodeField` of per-component 0/1 flags (see
        `mask`), not a boolean. Combine masks with `*` and use them to weight or
        select values."""
    def __gt__(self, other: float) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`field > x` → a 0/1 mask field (see `__ge__`)."""
    def __le__(self, other: float) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`field <= x` → a 0/1 mask field (see `__ge__`)."""
    def __lt__(self, other: float) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`field < x` → a 0/1 mask field (see `__ge__`)."""
    "# }
}

#[cfg(feature = "stub-gen")]
pyo3_stub_gen::inventory::submit! {
    pyo3_stub_gen::derive::gen_methods_from_python! { r#"
class PySubNodeField:
    @overload
    def __getitem__(self, key: tuple[pyo3_stub_gen.RustType["PyNode"], str]) -> float:
        """`subfield[node, "UX"]` → the value carried by that node for that
        component. Raises if either is absent from this zone."""
    @overload
    def __getitem__(self, key: str) -> pyo3_stub_gen.RustType["PySubNodeField"]:
        """`subfield["UX"]` → a fresh `SubNodeField` on the same support,
        keeping only that component."""
    @overload
    def __getitem__(self, key: list[str]) -> pyo3_stub_gen.RustType["PySubNodeField"]:
        """`subfield[["UX", "UY"]]` → a fresh `SubNodeField` keeping only those
        components, so `u1[u2.components()]` reprojects `u1` onto `u2`'s
        component set."""
    def __ge__(self, other: float) -> pyo3_stub_gen.RustType["PySubNodeField"]:
        """`subfield >= x` → a fresh `SubNodeField` of per-component 0/1 flags
        (see `mask`), not a boolean."""
    def __gt__(self, other: float) -> pyo3_stub_gen.RustType["PySubNodeField"]:
        """`subfield > x` → a 0/1 mask field (see `__ge__`)."""
    def __le__(self, other: float) -> pyo3_stub_gen.RustType["PySubNodeField"]:
        """`subfield <= x` → a 0/1 mask field (see `__ge__`)."""
    def __lt__(self, other: float) -> pyo3_stub_gen.RustType["PySubNodeField"]:
        """`subfield < x` → a 0/1 mask field (see `__ge__`)."""
    "# }
}

crate::impl_aggregate_pymethods!(
    PyNodeField,
    PySubNodeField,
    "NodeField",
    subfield,
    NodeField,
    r#"
class PyNodeField:
    @overload
    def __getitem__(self, key: int) -> pyo3_stub_gen.RustType["PySubNodeField"]:
        """`field[i]` → the `SubNodeField` of zone i: the values carried by the
        nodes of one support."""
    @overload
    def __getitem__(self, key: slice) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`field[i:j:k]` → a fresh `NodeField` over the sliced zones, shared
        with this one (no deep copy)."""
    @overload
    def __getitem__(self, key: str) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`field["u_x"]` → a fresh `NodeField` keeping only that component, on
        every zone that carries it."""
    @overload
    def __getitem__(self, key: list[str]) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`field[["u_x", "u_y"]]` → a fresh `NodeField` keeping only those
        components, on every zone that carries them."""
    def __or__(self, other: pyo3_stub_gen.RustType["PyNodeField"] | pyo3_stub_gen.RustType["PySubNodeField"]) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`field | other` → a fresh `NodeField` holding the zones of both.
        Zones sharing the same support are **fused** (union of their
        components) — unlike `ElementField`, which juxtaposes them."""
    def __ror__(self, other: pyo3_stub_gen.RustType["PySubNodeField"]) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`subfield | field` — the mirror of `field | subfield`, differing only
        in that the lone zone comes first."""
    "#,
    r#"
class PySubNodeField:
    def __or__(self, other: pyo3_stub_gen.RustType["PySubNodeField"]) -> pyo3_stub_gen.RustType["PyNodeField"]:
        """`subfield | subfield` → a fresh `NodeField` holding both zones, fused
        if they share the same support."""
    "#,
    field_components
);
