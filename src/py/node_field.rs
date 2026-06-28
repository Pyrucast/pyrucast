//! Python wrappers for [`crate::containers::node_field::SubNodeField`] and
//! [`crate::containers::node_field::NodeField`].

use crate::aggregate::Aggregate;
use crate::containers::field::SubField;
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::py::mesh::{PyMesh, PySubMesh};
use crate::py::node::PyNode;
use crate::store::{insert, read, write, Handle};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

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
        Ok(read(&self.handle)?.node_count())
    }

    /// Number of components stored per node.
    fn component_count(&self) -> PyResult<usize> {
        Ok(read(&self.handle)?.component_count())
    }

    /// Component names, in order.
    fn components(&self) -> PyResult<Vec<String>> {
        Ok(read(&self.handle)?.components().to_vec())
    }

    /// Value at node index `node_idx`, component index `comp_idx`.
    fn get(&self, node_idx: usize, comp_idx: usize) -> PyResult<f64> {
        Ok(read(&self.handle)?.get(node_idx, comp_idx)?)
    }

    /// Set the value at node index `node_idx`, component index `comp_idx`.
    fn set(&self, node_idx: usize, comp_idx: usize, value: f64) -> PyResult<()> {
        write(&self.handle)?.set(node_idx, comp_idx, value)?;
        Ok(())
    }

    /// Value at `node`, component index `comp_idx`.
    fn get_by_node(&self, node: PyRef<'_, PyNode>, comp_idx: usize) -> PyResult<f64> {
        let nid = node.as_node().id();
        Ok(read(&self.handle)?.get_by_node(nid, comp_idx)?)
    }

    /// Set the value at `node`, component index `comp_idx`.
    fn set_by_node(&self, node: PyRef<'_, PyNode>, comp_idx: usize, value: f64) -> PyResult<()> {
        let nid = node.as_node().id();
        write(&self.handle)?.set_by_node(nid, comp_idx, value)?;
        Ok(())
    }

    /// Index of component `name`, or `None` if unknown.
    fn component_index(&self, name: &str) -> PyResult<Option<usize>> {
        Ok(read(&self.handle)?.component_index(name))
    }

    /// All component values at node index `node_idx`, in order.
    fn node_values(&self, node_idx: usize) -> PyResult<Vec<f64>> {
        Ok(read(&self.handle)?.node_values(node_idx)?.to_vec())
    }

    /// The POI1 `SubMesh` this sub-field is supported on.
    fn support_submesh(&self) -> PyResult<PySubMesh> {
        let sm = read(&self.handle)?.support_submesh()?;
        Ok(PySubMesh { handle: insert(sm) })
    }

    /// The POI1 `Mesh` this sub-field is supported on.
    fn support_mesh(&self) -> PyResult<PyMesh> {
        let mesh = read(&self.handle)?.support_mesh()?;
        Ok(PyMesh { inner: mesh })
    }

    /// Value at `node` for the named `component`.
    fn value(&self, node: PyRef<'_, PyNode>, component: &str) -> PyResult<f64> {
        let nid = node.as_node().id();
        Ok(read(&self.handle)?.value(nid, component)?)
    }

    /// Set the value at `node` for the named `component`.
    fn set_value(&self, node: PyRef<'_, PyNode>, component: &str, value: f64) -> PyResult<()> {
        let nid = node.as_node().id();
        write(&self.handle)?.set_value(nid, component, value)?;
        Ok(())
    }

    /// Smallest value of the named `component`.
    fn min(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::min(&*read(&self.handle)?, component)?)
    }

    /// Largest value of the named `component`.
    fn max(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::SubField;
        Ok(SubField::max(&*read(&self.handle)?, component)?)
    }

    /// Add `scalar` to every value of `component` (in place).
    fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        write(&self.handle)?.add_to_component(component, scalar)?;
        Ok(())
    }

    /// Subtract `scalar` from every value of `component` (in place).
    fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        write(&self.handle)?.sub_to_component(component, scalar)?;
        Ok(())
    }

    /// Multiply every value of `component` by `scalar` (in place).
    fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        write(&self.handle)?.mul_to_component(component, scalar)?;
        Ok(())
    }

    /// Divide every value of `component` by `scalar` (in place).
    fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
        write(&self.handle)?.div_to_component(component, scalar)?;
        Ok(())
    }

    // ── Arithmetic operators (return a new sub-field) ───────────────────
    //
    // `rhs` may be a float (scalar broadcast over every node × component) or
    // another `SubNodeField` (element-by-element, strict: same support and
    // same components). Division does not guard against zero (inf/nan).

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

    /// `subfield[node, "UX"]` — raises if the node or component is absent.
    fn __getitem__(&self, key: (PyRef<'_, PyNode>, String)) -> PyResult<f64> {
        let (node, comp) = key;
        let nid = node.as_node().id();
        Ok(read(&self.handle)?.value(nid, &comp)?)
    }

    /// `subfield[node, "UX"] = v` — raises if the node or component is absent.
    fn __setitem__(&self, key: (PyRef<'_, PyNode>, String), value: f64) -> PyResult<()> {
        let (node, comp) = key;
        let nid = node.as_node().id();
        write(&self.handle)?.set_value(nid, &comp, value)?;
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", &*read(&self.handle)?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", &*read(&self.handle)?))
    }
}

impl PySubNodeField {
    /// Dispatch an arithmetic operator: float → scalar broadcast,
    /// `SubNodeField` → strict element-by-element `combine`.
    fn scalar_or_combine(
        &self,
        rhs: &Bound<'_, PyAny>,
        op: fn(f64, f64) -> f64,
    ) -> PyResult<PySubNodeField> {
        if let Ok(s) = rhs.extract::<f64>() {
            let out = read(&self.handle)?.map_all(|v| op(v, s));
            Ok(PySubNodeField {
                handle: insert(out),
            })
        } else if let Ok(other) = rhs.extract::<PyRef<PySubNodeField>>() {
            let a = (*read(&self.handle)?).clone();
            let b = (*read(&other.handle)?).clone();
            Ok(PySubNodeField {
                handle: insert(a.combine(&b, op)?),
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

    /// Verify zone coherence: every `(node, component)` stored by several
    /// zones must hold the same value everywhere. Raises on the first
    /// conflict.
    fn check(&self) -> PyResult<()> {
        Ok(self.inner.check()?)
    }

    /// Smallest value of `component` across the zones defining it.
    fn min(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::min(&self.inner, component)?)
    }

    /// Largest value of `component` across the zones defining it.
    fn max(&self, component: &str) -> PyResult<f64> {
        use crate::containers::field::Field;
        Ok(Field::max(&self.inner, component)?)
    }

    /// Visualize this field alone, as a **coloured point cloud** over
    /// its support nodes — the POI1 support has no connectivity, so no
    /// surface can be drawn here; use `mesh.plot(field=...)` with the
    /// original mesh for surfaces.
    ///
    /// Same `view` / `save` / `show_axes` / `component` / `vmin` /
    /// `vmax` / `cmap` semantics as `Mesh.plot`.
    #[cfg(feature = "viz")]
    #[pyo3(signature = (view=None, save=None, show_axes=true, component=None, vmin=None, vmax=None, cmap=None))]
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
    ) -> PyResult<()> {
        let mut view = view
            .map(|(yaw, pitch, scale)| crate::viz::View {
                yaw,
                pitch,
                scale,
                target: None,
                show_axes,
            })
            .unwrap_or_else(crate::viz::View::default);
        view.show_axes = show_axes;
        let scale = crate::viz::ColorScale {
            cmap: crate::py::mesh::parse_cmap(cmap)?,
            vmin,
            vmax,
        };
        self.inner
            .plot(Some(view), save.as_deref(), component.as_deref(), scale)?;
        Ok(())
    }

    /// A `Mesh` mirroring this field's supports — the zones' POI1
    /// support submeshes, shared (not copied).
    fn support_mesh(&self) -> PyResult<PyMesh> {
        let mut mesh = crate::containers::mesh::Mesh::empty();
        for h in &self.inner {
            let sm = read(h)?.support();
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
    /// `combine_field` (same decomposition), `SubNodeField` →
    /// `combine_subfield` (targeted zone update).
    fn binary(&self, rhs: &Bound<'_, PyAny>, op: fn(f64, f64) -> f64) -> PyResult<PyNodeField> {
        use crate::containers::field::Field;
        if let Ok(s) = rhs.extract::<f64>() {
            Ok(PyNodeField {
                inner: self.inner.combine_scalar(op, s)?,
            })
        } else if let Ok(other) = rhs.extract::<PyRef<PyNodeField>>() {
            Ok(PyNodeField {
                inner: self.inner.combine_field(&other.inner, op)?,
            })
        } else if let Ok(sub) = rhs.extract::<PyRef<PySubNodeField>>() {
            let s = (*read(&sub.handle)?).clone();
            Ok(PyNodeField {
                inner: self.inner.combine_subfield(&s, op)?,
            })
        } else {
            Err(PyTypeError::new_err(
                "unsupported operand: expected a float, a NodeField, or a SubNodeField",
            ))
        }
    }
}

crate::impl_aggregate_pymethods!(
    PyNodeField,
    PySubNodeField,
    "NodeField",
    subfield,
    NodeField
);
