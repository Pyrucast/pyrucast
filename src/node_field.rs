//! NodeField — multi-component values on a POI1 submesh.
//!
//! A [`NodeField`] stores one or more named components per node of a
//! support defined by a POI1 [`SubMesh`] (a list of nodes). The set of
//! nodes is captured **at construction** as a snapshot, and remains
//! stable for the lifetime of the field: cells added to the originating
//! POI1 SubMesh after construction do not affect a previously created
//! field. Each node in the support is increfed in the
//! [`Configuration`]; the field's `Drop` decrefs them all.
//!
//! The default value of every component is `0.0`.
//!
//! # Example
//!
//! ```
//! use pyrucast::configuration::Configuration;
//! use pyrucast::element_type::ElementType;
//! use pyrucast::mesh::SubMesh;
//! use pyrucast::node::Node;
//! use pyrucast::node_field::NodeField;
//! use pyrucast::store::{insert, with, with_mut};
//!
//! let cfg = insert(Configuration::new(2).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
//!
//! // Build a POI1 SubMesh holding [a, b], then a 2-component field on it.
//! let sm_handle = {
//!     let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
//!     sm.add_cell(&[a.id()]).unwrap();
//!     sm.add_cell(&[b.id()]).unwrap();
//!     insert(sm)
//! };
//!
//! let mut field = NodeField::from_poi1(
//!     &sm_handle,
//!     vec!["UX".into(), "UY".into()],
//! ).unwrap();
//! field.set(0, 0, 1.5).unwrap();
//! field.set(0, 1, -0.25).unwrap();
//! assert_eq!(field.get(0, 0).unwrap(), 1.5);
//! assert_eq!(field.get(0, 1).unwrap(), -0.25);
//! assert_eq!(field.get(1, 0).unwrap(), 0.0);  // default
//! ```

use crate::configuration::{Configuration, NodeId};
use crate::element_type::ElementType;
use crate::error::{PyrucastError, Result};
use crate::mesh::{Mesh, SubMesh};
use crate::store::{insert, with, with_mut, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

// ─── NodeField ──────────────────────────────────────────────────────────────

/// Multi-component values on a snapshot of a POI1 SubMesh's node list.
///
/// Values are stored row-major: component `c` of node `i` is at index
/// `i * component_count + c` in the internal flat buffer.
#[derive(Serialize, Deserialize)]
pub struct NodeField {
    cfg: Handle<Configuration>,
    /// Snapshot of the support. Each id holds one incref in the Configuration.
    nodes: Vec<NodeId>,
    components: Vec<String>,
    /// Row-major: `values[i * components.len() + c]`.
    values: Vec<f64>,
}

impl NodeField {
    /// Build a NodeField on the nodes of a POI1 [`SubMesh`]. The support
    /// is captured as a snapshot; subsequent changes to the SubMesh do
    /// not affect this field.
    ///
    /// Errors:
    /// - the SubMesh is not POI1,
    /// - `components` is empty,
    /// - `components` contains duplicate names.
    pub fn from_poi1(submesh: &Handle<SubMesh>, components: Vec<String>) -> Result<Self> {
        if components.is_empty() {
            return Err(PyrucastError::Message(
                "NodeField requires at least one component".into(),
            ));
        }
        for i in 0..components.len() {
            for j in (i + 1)..components.len() {
                if components[i] == components[j] {
                    return Err(PyrucastError::Message(format!(
                        "duplicate component name: {}",
                        components[i]
                    )));
                }
            }
        }

        let (cfg, nodes) = with(submesh, |sm| -> Result<_> {
            if sm.element_type() != ElementType::POI1 {
                return Err(PyrucastError::Message(format!(
                    "NodeField requires a POI1 SubMesh, got {}",
                    sm.element_type()
                )));
            }
            let cfg = sm.configuration();
            // POI1: connectivity is exactly the node list (1 node per cell).
            let nodes: Vec<NodeId> = sm.connectivity().to_vec();
            Ok((cfg, nodes))
        })??;

        // Acquire one incref per node, rolling back on partial failure.
        let result: Result<()> = with_mut(&cfg, |c| {
            let mut acquired = 0usize;
            for &nid in &nodes {
                if let Err(e) = c.incref(nid) {
                    for &m in &nodes[..acquired] {
                        let _ = c.decref(m);
                    }
                    return Err(e);
                }
                acquired += 1;
            }
            Ok(())
        })?;
        result?;

        let n_nodes = nodes.len();
        let n_comp = components.len();
        let values = vec![0.0; n_nodes * n_comp];
        Ok(NodeField {
            cfg,
            nodes,
            components,
            values,
        })
    }

    /// Number of nodes in the support.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of components.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Component names, in order.
    pub fn components(&self) -> &[String] {
        &self.components
    }

    /// Handle to the owning `Configuration` (internal clone).
    pub fn configuration(&self) -> Handle<Configuration> {
        self.cfg.clone()
    }

    /// Read a value by `(node_index, component_index)`.
    pub fn get(&self, node_idx: usize, comp_idx: usize) -> Result<f64> {
        self.check_indices(node_idx, comp_idx)?;
        Ok(self.values[node_idx * self.components.len() + comp_idx])
    }

    /// Write a value by `(node_index, component_index)`.
    pub fn set(&mut self, node_idx: usize, comp_idx: usize, value: f64) -> Result<()> {
        self.check_indices(node_idx, comp_idx)?;
        let ncomp = self.components.len();
        self.values[node_idx * ncomp + comp_idx] = value;
        Ok(())
    }

    /// Read a value by `(NodeId, component_index)`.
    pub fn get_by_node(&self, nid: NodeId, comp_idx: usize) -> Result<f64> {
        let i = self.index_of(nid).ok_or_else(|| {
            PyrucastError::Message(format!("node {} not in field support", nid))
        })?;
        self.get(i, comp_idx)
    }

    /// Write a value by `(NodeId, component_index)`.
    pub fn set_by_node(&mut self, nid: NodeId, comp_idx: usize, value: f64) -> Result<()> {
        let i = self.index_of(nid).ok_or_else(|| {
            PyrucastError::Message(format!("node {} not in field support", nid))
        })?;
        self.set(i, comp_idx, value)
    }

    /// Position of a NodeId in the support, or `None` if absent.
    pub fn index_of(&self, nid: NodeId) -> Option<usize> {
        self.nodes.iter().position(|&n| n == nid)
    }

    /// Index of a named component, or `None` if absent.
    pub fn component_index(&self, name: &str) -> Option<usize> {
        self.components.iter().position(|c| c == name)
    }

    /// Build a new POI1 [`SubMesh`] whose cells are exactly the support nodes
    /// of this field, in the same order. Each node is increfed by the new
    /// submesh independently of this field's own increfs.
    ///
    /// # Example
    ///
    /// ```
    /// use pyrucast::configuration::Configuration;
    /// use pyrucast::element_type::ElementType;
    /// use pyrucast::mesh::{Mesh, SubMesh};
    /// use pyrucast::node::Node;
    /// use pyrucast::node_field::NodeField;
    /// use pyrucast::store::{insert, with};
    ///
    /// let cfg = insert(Configuration::new(2).unwrap());
    /// let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
    /// let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
    ///
    /// let sm_handle = {
    ///     let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
    ///     sm.add_cell(&[a.id()]).unwrap();
    ///     sm.add_cell(&[b.id()]).unwrap();
    ///     insert(sm)
    /// };
    ///
    /// let field = NodeField::from_poi1(&sm_handle, vec!["T".into()]).unwrap();
    /// let sm2 = field.to_poi1_submesh().unwrap();
    /// assert_eq!(sm2.cell_count(), 2);
    /// // Verify node order via Mesh::node (public API).
    /// let mut m = Mesh::new(cfg.clone());
    /// m.add_submesh(insert(sm2)).unwrap();
    /// assert_eq!(m.node(0, 0, 0).unwrap().id(), a.id());
    /// assert_eq!(m.node(0, 1, 0).unwrap().id(), b.id());
    /// ```
    pub fn to_poi1_submesh(&self) -> Result<SubMesh> {
        let mut sm = SubMesh::new(self.cfg.clone(), ElementType::POI1);
        for &nid in &self.nodes {
            sm.add_cell(&[nid])?;
        }
        Ok(sm)
    }

    /// Build a [`Mesh`] with a single POI1 submesh mirroring the support of
    /// this field.
    pub fn to_poi1_mesh(&self) -> Result<Mesh> {
        let sm_handle = insert(self.to_poi1_submesh()?);
        let mut mesh = Mesh::new(self.cfg.clone());
        mesh.add_submesh(sm_handle)?;
        Ok(mesh)
    }

    /// All values of node `node_idx`, in component order.
    pub fn node_values(&self, node_idx: usize) -> Result<&[f64]> {
        if node_idx >= self.nodes.len() {
            return Err(PyrucastError::Message(format!(
                "node index {} ≥ node_count {}",
                node_idx,
                self.nodes.len()
            )));
        }
        let ncomp = self.components.len();
        Ok(&self.values[node_idx * ncomp..(node_idx + 1) * ncomp])
    }

    // ── Constructeur interne ────────────────────────────────────────────────

    /// Builds a NodeField from an explicit node list, with all values at 0.0.
    /// Increfs every node; on partial failure the acquired increfs are rolled back.
    fn new_with_nodes(
        cfg: Handle<Configuration>,
        nodes: Vec<NodeId>,
        components: Vec<String>,
    ) -> Result<Self> {
        let n_nodes = nodes.len();
        let n_comp = components.len();
        let result: Result<()> = with_mut(&cfg, |c| {
            let mut acquired = 0usize;
            for &nid in &nodes {
                if let Err(e) = c.incref(nid) {
                    for &m in &nodes[..acquired] {
                        let _ = c.decref(m);
                    }
                    return Err(e);
                }
                acquired += 1;
            }
            Ok(())
        })?;
        result?;
        Ok(NodeField {
            cfg,
            nodes,
            components,
            values: vec![0.0; n_nodes * n_comp],
        })
    }

    // ── Helpers privés ──────────────────────────────────────────────────────

    fn component_value_opt(&self, nid: NodeId, comp: &str) -> Option<f64> {
        let ni = self.index_of(nid)?;
        let ci = self.component_index(comp)?;
        Some(self.values[ni * self.components.len() + ci])
    }

    fn get_or_default(&self, nid: NodeId, comp: &str) -> f64 {
        self.component_value_opt(nid, comp).unwrap_or(0.0)
    }

    fn check_compatible(&self, other: &NodeField) -> Result<()> {
        if self.cfg.index() != other.cfg.index()
            || self.cfg.generation() != other.cfg.generation()
        {
            return Err(PyrucastError::Message(
                "fields are not attached to the same Configuration".into(),
            ));
        }
        Ok(())
    }

    /// Returns (union_components, union_nodes): self's items first, then other's extras.
    fn union_layout(&self, other: &NodeField) -> (Vec<String>, Vec<NodeId>) {
        let mut components = self.components.clone();
        for c in &other.components {
            if !components.iter().any(|x| x == c) {
                components.push(c.clone());
            }
        }
        let mut nodes = self.nodes.clone();
        for &nid in &other.nodes {
            if !nodes.contains(&nid) {
                nodes.push(nid);
            }
        }
        (components, nodes)
    }

    // ── Accès idiomatique ───────────────────────────────────────────────────

    /// Read a value by `(NodeId, component name)`.
    pub fn value(&self, nid: NodeId, component: &str) -> Result<f64> {
        let ni = self.index_of(nid).ok_or_else(|| {
            PyrucastError::Message(format!("node {} not in field support", nid))
        })?;
        let ci = self.component_index(component).ok_or_else(|| {
            PyrucastError::Message(format!("unknown component: {}", component))
        })?;
        Ok(self.values[ni * self.components.len() + ci])
    }

    /// Write a value by `(NodeId, component name)`.
    pub fn set_value(&mut self, nid: NodeId, component: &str, value: f64) -> Result<()> {
        let ni = self.index_of(nid).ok_or_else(|| {
            PyrucastError::Message(format!("node {} not in field support", nid))
        })?;
        let ci = self.component_index(component).ok_or_else(|| {
            PyrucastError::Message(format!("unknown component: {}", component))
        })?;
        let ncomp = self.components.len();
        self.values[ni * ncomp + ci] = value;
        Ok(())
    }

    // ── Opérations entre champs ─────────────────────────────────────────────

    /// Component-wise addition of two fields.
    ///
    /// Missing nodes or components in either field are treated as `0.0`.
    /// Both fields must be attached to the same `Configuration`.
    /// The result's component list is the union of both fields' (self first),
    /// and similarly for nodes.
    pub fn add_fields(&self, other: &NodeField) -> Result<NodeField> {
        self.check_compatible(other)?;
        let (components, nodes) = self.union_layout(other);
        let mut result =
            NodeField::new_with_nodes(self.cfg.clone(), nodes.clone(), components.clone())?;
        let ncomp = components.len();
        for (ni, &nid) in nodes.iter().enumerate() {
            for (ci, comp) in components.iter().enumerate() {
                result.values[ni * ncomp + ci] =
                    self.get_or_default(nid, comp) + other.get_or_default(nid, comp);
            }
        }
        Ok(result)
    }

    /// Merge two fields.
    ///
    /// Like [`add_fields`](Self::add_fields) but errors if both fields have
    /// a **different** value at the same `(node, component)` pair. Equal
    /// values at shared points are kept as-is.
    pub fn merge_fields(&self, other: &NodeField) -> Result<NodeField> {
        self.check_compatible(other)?;
        let (components, nodes) = self.union_layout(other);
        let mut result =
            NodeField::new_with_nodes(self.cfg.clone(), nodes.clone(), components.clone())?;
        let ncomp = components.len();
        for (ni, &nid) in nodes.iter().enumerate() {
            for (ci, comp) in components.iter().enumerate() {
                let va = self.component_value_opt(nid, comp);
                let vb = other.component_value_opt(nid, comp);
                let v = match (va, vb) {
                    (None, None) => 0.0,
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (Some(a), Some(b)) if a == b => a,
                    (Some(a), Some(b)) => {
                        return Err(PyrucastError::Message(format!(
                            "merge_fields: conflicting values at node {}, \
                             component \"{}\": {} vs {}",
                            nid, comp, a, b
                        )))
                    }
                };
                result.values[ni * ncomp + ci] = v;
            }
        }
        Ok(result)
    }

    // ── Opérations scalaires sur une composante ─────────────────────────────

    /// Add `scalar` to every node's value for the named component.
    pub fn add_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        let ci = self.component_index(component).ok_or_else(|| {
            PyrucastError::Message(format!("unknown component: {}", component))
        })?;
        let ncomp = self.components.len();
        for i in 0..self.nodes.len() {
            self.values[i * ncomp + ci] += scalar;
        }
        Ok(())
    }

    /// Subtract `scalar` from every node's value for the named component.
    pub fn sub_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        let ci = self.component_index(component).ok_or_else(|| {
            PyrucastError::Message(format!("unknown component: {}", component))
        })?;
        let ncomp = self.components.len();
        for i in 0..self.nodes.len() {
            self.values[i * ncomp + ci] -= scalar;
        }
        Ok(())
    }

    /// Multiply every node's value for the named component by `scalar`.
    pub fn mul_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        let ci = self.component_index(component).ok_or_else(|| {
            PyrucastError::Message(format!("unknown component: {}", component))
        })?;
        let ncomp = self.components.len();
        for i in 0..self.nodes.len() {
            self.values[i * ncomp + ci] *= scalar;
        }
        Ok(())
    }

    /// Divide every node's value for the named component by `scalar`.
    ///
    /// Returns an error if `scalar` is zero.
    pub fn div_to_component(&mut self, component: &str, scalar: f64) -> Result<()> {
        if scalar == 0.0 {
            return Err(PyrucastError::Message(
                "div_to_component: division by zero".into(),
            ));
        }
        let ci = self.component_index(component).ok_or_else(|| {
            PyrucastError::Message(format!("unknown component: {}", component))
        })?;
        let ncomp = self.components.len();
        for i in 0..self.nodes.len() {
            self.values[i * ncomp + ci] /= scalar;
        }
        Ok(())
    }

    // ── Réduction sur maillage ──────────────────────────────────────────────

    /// Restrict this field to the nodes of `mesh`.
    ///
    /// Returns a new field with the same components, defined on the unique
    /// nodes of `mesh` in order of first appearance. Nodes of `mesh` absent
    /// from this field are assigned `0.0`. The mesh must be attached to the
    /// same `Configuration` as this field.
    pub fn restrict(&self, mesh: &Mesh) -> Result<NodeField> {
        let mesh_cfg = mesh.configuration();
        if mesh_cfg.index() != self.cfg.index() || mesh_cfg.generation() != self.cfg.generation() {
            return Err(PyrucastError::Message(
                "restrict: mesh is not attached to the same Configuration".into(),
            ));
        }
        let mut mesh_nodes: Vec<NodeId> = Vec::new();
        for i in 0..mesh.submesh_count() {
            let sm_handle = mesh.submesh(i)?;
            let connectivity = with(&sm_handle, |sm| sm.connectivity().to_vec())?;
            for nid in connectivity {
                if !mesh_nodes.contains(&nid) {
                    mesh_nodes.push(nid);
                }
            }
        }
        let ncomp = self.components.len();
        let mut result =
            NodeField::new_with_nodes(self.cfg.clone(), mesh_nodes, self.components.clone())?;
        for (ni, &nid) in result.nodes.iter().enumerate() {
            if let Some(self_ni) = self.index_of(nid) {
                let src = self_ni * ncomp;
                let dst = ni * ncomp;
                result.values[dst..dst + ncomp].copy_from_slice(&self.values[src..src + ncomp]);
            }
        }
        Ok(result)
    }

    fn check_indices(&self, ni: usize, ci: usize) -> Result<()> {
        if ni >= self.nodes.len() {
            return Err(PyrucastError::Message(format!(
                "node index {} ≥ node_count {}",
                ni,
                self.nodes.len()
            )));
        }
        if ci >= self.components.len() {
            return Err(PyrucastError::Message(format!(
                "component index {} ≥ component_count {}",
                ci,
                self.components.len()
            )));
        }
        Ok(())
    }
}

impl Drop for NodeField {
    fn drop(&mut self) {
        // One lock acquisition for all decrefs.
        let _ = with_mut(&self.cfg, |c| {
            for &n in &self.nodes {
                let _ = c.decref(n);
            }
        });
    }
}

impl fmt::Debug for NodeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeField")
            .field("node_count", &self.nodes.len())
            .field("components", &self.components)
            .finish()
    }
}

impl fmt::Display for NodeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NodeField: {} node(s), {} component(s) [{}]",
            self.nodes.len(),
            self.components.len(),
            self.components.join(", ")
        )
    }
}

// ─── Index ──────────────────────────────────────────────────────────────────

/// `field[(nid, "UX")]` — panics if the node or component is absent.
impl Index<(NodeId, &str)> for NodeField {
    type Output = f64;
    fn index(&self, (nid, comp): (NodeId, &str)) -> &f64 {
        let ni = self
            .index_of(nid)
            .unwrap_or_else(|| panic!("node {} not in field support", nid));
        let ci = self
            .component_index(comp)
            .unwrap_or_else(|| panic!("unknown component: {}", comp));
        &self.values[ni * self.components.len() + ci]
    }
}

/// `field[(nid, "UX")] = v` — panics if the node or component is absent.
impl IndexMut<(NodeId, &str)> for NodeField {
    fn index_mut(&mut self, (nid, comp): (NodeId, &str)) -> &mut f64 {
        let ni = self
            .index_of(nid)
            .unwrap_or_else(|| panic!("node {} not in field support", nid));
        let ci = self
            .component_index(comp)
            .unwrap_or_else(|| panic!("unknown component: {}", comp));
        let ncomp = self.components.len();
        &mut self.values[ni * ncomp + ci]
    }
}

// ─── Clone ──────────────────────────────────────────────────────────────────

impl Clone for NodeField {
    fn clone(&self) -> Self {
        // Self holds an incref on every node, so they are guaranteed alive.
        with_mut(&self.cfg, |c| {
            for &nid in &self.nodes {
                let _ = c.incref(nid);
            }
        })
        .unwrap();
        NodeField {
            cfg: self.cfg.clone(),
            nodes: self.nodes.clone(),
            components: self.components.clone(),
            values: self.values.clone(),
        }
    }
}

// ─── Opérateurs field OP f64 ────────────────────────────────────────────────
//
// Consuming versions (modify in place, return self) are zero-copy.
// Reference versions clone first.

impl Add<f64> for NodeField {
    type Output = NodeField;
    fn add(mut self, rhs: f64) -> NodeField {
        for v in &mut self.values {
            *v += rhs;
        }
        self
    }
}

impl Add<f64> for &NodeField {
    type Output = NodeField;
    fn add(self, rhs: f64) -> NodeField {
        let mut result = self.clone();
        for v in &mut result.values {
            *v += rhs;
        }
        result
    }
}

impl Sub<f64> for NodeField {
    type Output = NodeField;
    fn sub(mut self, rhs: f64) -> NodeField {
        for v in &mut self.values {
            *v -= rhs;
        }
        self
    }
}

impl Sub<f64> for &NodeField {
    type Output = NodeField;
    fn sub(self, rhs: f64) -> NodeField {
        let mut result = self.clone();
        for v in &mut result.values {
            *v -= rhs;
        }
        result
    }
}

impl Mul<f64> for NodeField {
    type Output = NodeField;
    fn mul(mut self, rhs: f64) -> NodeField {
        for v in &mut self.values {
            *v *= rhs;
        }
        self
    }
}

impl Mul<f64> for &NodeField {
    type Output = NodeField;
    fn mul(self, rhs: f64) -> NodeField {
        let mut result = self.clone();
        for v in &mut result.values {
            *v *= rhs;
        }
        result
    }
}

impl Div<f64> for NodeField {
    type Output = NodeField;
    fn div(mut self, rhs: f64) -> NodeField {
        for v in &mut self.values {
            *v /= rhs;
        }
        self
    }
}

impl Div<f64> for &NodeField {
    type Output = NodeField;
    fn div(self, rhs: f64) -> NodeField {
        let mut result = self.clone();
        for v in &mut result.values {
            *v /= rhs;
        }
        result
    }
}

// ─── Python binding ─────────────────────────────────────────────────────────

#[cfg(feature = "extension-module")]
mod python {
    use super::*;
    use crate::mesh::{PyMesh, PySubMesh};
    use crate::store::insert;
    use pyo3::prelude::*;

    /// Python wrapper for [`NodeField`].
    #[pyclass(name = "NodeField")]
    pub struct PyNodeField {
        handle: Handle<NodeField>,
    }

    #[pymethods]
    impl PyNodeField {
        #[new]
        fn py_new(submesh: PyRef<PySubMesh>, components: Vec<String>) -> PyResult<Self> {
            let sm_handle = submesh.handle.clone();
            let nf = NodeField::from_poi1(&sm_handle, components)?;
            Ok(Self {
                handle: insert(nf),
            })
        }

        fn node_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |f| f.node_count())?)
        }

        fn component_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |f| f.component_count())?)
        }

        fn components(&self) -> PyResult<Vec<String>> {
            Ok(with(&self.handle, |f| f.components().to_vec())?)
        }

        fn get(&self, node_idx: usize, comp_idx: usize) -> PyResult<f64> {
            Ok(with(&self.handle, |f| f.get(node_idx, comp_idx))??)
        }

        fn set(&self, node_idx: usize, comp_idx: usize, value: f64) -> PyResult<()> {
            with_mut(&self.handle, |f| f.set(node_idx, comp_idx, value))??;
            Ok(())
        }

        fn get_by_node(&self, node_id: u32, comp_idx: usize) -> PyResult<f64> {
            Ok(with(&self.handle, |f| f.get_by_node(NodeId(node_id), comp_idx))??)
        }

        fn set_by_node(&self, node_id: u32, comp_idx: usize, value: f64) -> PyResult<()> {
            with_mut(&self.handle, |f| {
                f.set_by_node(NodeId(node_id), comp_idx, value)
            })??;
            Ok(())
        }

        fn component_index(&self, name: &str) -> PyResult<Option<usize>> {
            Ok(with(&self.handle, |f| f.component_index(name))?)
        }

        fn node_values(&self, node_idx: usize) -> PyResult<Vec<f64>> {
            Ok(with(&self.handle, |f| f.node_values(node_idx).map(|s| s.to_vec()))??)
        }

        fn to_poi1_submesh(&self) -> PyResult<PySubMesh> {
            let sm = with(&self.handle, |f| f.to_poi1_submesh())??;
            Ok(PySubMesh { handle: insert(sm) })
        }

        fn to_poi1_mesh(&self) -> PyResult<PyMesh> {
            let mesh = with(&self.handle, |f| f.to_poi1_mesh())??;
            Ok(PyMesh { handle: insert(mesh) })
        }

        fn value(&self, node_id: u32, component: &str) -> PyResult<f64> {
            Ok(with(&self.handle, |f| f.value(NodeId(node_id), component))??)
        }

        fn set_value(&self, node_id: u32, component: &str, value: f64) -> PyResult<()> {
            with_mut(&self.handle, |f| f.set_value(NodeId(node_id), component, value))??;
            Ok(())
        }

        fn add_fields(&self, other: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
            let result = with(&self.handle, |a| {
                with(&other.handle, |b| a.add_fields(b))?
            })??;
            Ok(PyNodeField {
                handle: insert(result),
            })
        }

        fn merge_fields(&self, other: PyRef<PyNodeField>) -> PyResult<PyNodeField> {
            let result = with(&self.handle, |a| {
                with(&other.handle, |b| a.merge_fields(b))?
            })??;
            Ok(PyNodeField {
                handle: insert(result),
            })
        }

        fn add_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
            with_mut(&self.handle, |f| f.add_to_component(component, scalar))??;
            Ok(())
        }

        fn sub_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
            with_mut(&self.handle, |f| f.sub_to_component(component, scalar))??;
            Ok(())
        }

        fn mul_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
            with_mut(&self.handle, |f| f.mul_to_component(component, scalar))??;
            Ok(())
        }

        fn div_to_component(&self, component: &str, scalar: f64) -> PyResult<()> {
            with_mut(&self.handle, |f| f.div_to_component(component, scalar))??;
            Ok(())
        }

        fn restrict(&self, mesh: PyRef<PyMesh>) -> PyResult<PyNodeField> {
            let result = with(&self.handle, |nf| {
                with(&mesh.handle, |m| nf.restrict(m))?
            })??;
            Ok(PyNodeField {
                handle: insert(result),
            })
        }

        fn __add__(&self, rhs: f64) -> PyResult<PyNodeField> {
            let result = with(&self.handle, |f| f + rhs)?;
            Ok(PyNodeField {
                handle: insert(result),
            })
        }

        fn __sub__(&self, rhs: f64) -> PyResult<PyNodeField> {
            let result = with(&self.handle, |f| f - rhs)?;
            Ok(PyNodeField {
                handle: insert(result),
            })
        }

        fn __mul__(&self, rhs: f64) -> PyResult<PyNodeField> {
            let result = with(&self.handle, |f| f * rhs)?;
            Ok(PyNodeField {
                handle: insert(result),
            })
        }

        fn __truediv__(&self, rhs: f64) -> PyResult<PyNodeField> {
            let result = with(&self.handle, |f| f / rhs)?;
            Ok(PyNodeField {
                handle: insert(result),
            })
        }

        /// `field[node_id, "UX"]` — raises IndexError if absent.
        fn __getitem__(&self, key: (u32, String)) -> PyResult<f64> {
            let (node_id, comp) = key;
            Ok(with(&self.handle, |f| f.value(NodeId(node_id), &comp))??)
        }

        /// `field[node_id, "UX"] = v` — raises IndexError if absent.
        fn __setitem__(&self, key: (u32, String), value: f64) -> PyResult<()> {
            let (node_id, comp) = key;
            with_mut(&self.handle, |f| f.set_value(NodeId(node_id), &comp, value))??;
            Ok(())
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |f| format!("{:?}", f))?)
        }

        fn __str__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |f| format!("{}", f))?)
        }
    }
}

#[cfg(feature = "extension-module")]
pub use python::PyNodeField;

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::store::insert;

    fn make_poi1_with(n_nodes: usize) -> (Handle<Configuration>, Vec<Node>, Handle<SubMesh>) {
        let cfg = insert(Configuration::new(2).unwrap());
        let nodes: Vec<Node> = (0..n_nodes)
            .map(|i| Node::create_in(cfg.clone(), &[i as f64, 0.0]).unwrap())
            .collect();
        let sm_handle = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            for n in &nodes {
                sm.add_cell(&[n.id()]).unwrap();
            }
            insert(sm)
        };
        (cfg, nodes, sm_handle)
    }

    #[test]
    fn from_poi1_zero_initialized() {
        let (_cfg, _nodes, sm) = make_poi1_with(3);
        let f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert_eq!(f.node_count(), 3);
        assert_eq!(f.component_count(), 1);
        for i in 0..3 {
            assert_eq!(f.get(i, 0).unwrap(), 0.0);
        }
    }

    #[test]
    fn from_poi1_rejects_non_poi1() {
        let cfg = insert(Configuration::new(2).unwrap());
        let sm = insert(SubMesh::new(cfg, ElementType::SEG2));
        let err = NodeField::from_poi1(&sm, vec!["X".into()]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn from_poi1_rejects_empty_components() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let err = NodeField::from_poi1(&sm, vec![]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn from_poi1_rejects_duplicate_components() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let err = NodeField::from_poi1(&sm, vec!["UX".into(), "UX".into()]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn increfs_and_drop_decrefs() {
        let (cfg, nodes, sm) = make_poi1_with(2);
        // At this point each node has refcount = 2 (Node + SubMesh).
        with(&cfg, |c| {
            assert_eq!(c.refcount(nodes[0].id()), 2);
            assert_eq!(c.refcount(nodes[1].id()), 2);
        })
        .unwrap();
        let f = NodeField::from_poi1(&sm, vec!["P".into()]).unwrap();
        // After NodeField construction: +1 each = 3.
        with(&cfg, |c| {
            assert_eq!(c.refcount(nodes[0].id()), 3);
            assert_eq!(c.refcount(nodes[1].id()), 3);
        })
        .unwrap();
        drop(f);
        // Back to 2.
        with(&cfg, |c| {
            assert_eq!(c.refcount(nodes[0].id()), 2);
            assert_eq!(c.refcount(nodes[1].id()), 2);
        })
        .unwrap();
    }

    #[test]
    fn get_set_multi_component() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let mut f =
            NodeField::from_poi1(&sm, vec!["UX".into(), "UY".into(), "UZ".into()]).unwrap();
        f.set(0, 0, 1.0).unwrap();
        f.set(0, 1, 2.0).unwrap();
        f.set(0, 2, 3.0).unwrap();
        f.set(1, 1, -7.0).unwrap();
        assert_eq!(f.node_values(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(f.node_values(1).unwrap(), &[0.0, -7.0, 0.0]);
    }

    #[test]
    fn get_set_by_node_and_component_name() {
        let (_cfg, nodes, sm) = make_poi1_with(2);
        let mut f = NodeField::from_poi1(&sm, vec!["T".into(), "P".into()]).unwrap();
        let ci_p = f.component_index("P").unwrap();
        f.set_by_node(nodes[1].id(), ci_p, 42.0).unwrap();
        assert_eq!(f.get_by_node(nodes[1].id(), ci_p).unwrap(), 42.0);
    }

    #[test]
    fn out_of_bounds_errors() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let mut f = NodeField::from_poi1(&sm, vec!["X".into()]).unwrap();
        assert!(f.get(5, 0).is_err());
        assert!(f.get(0, 5).is_err());
        assert!(f.set(5, 0, 1.0).is_err());
        assert!(f.node_values(5).is_err());
    }

    #[test]
    fn unknown_node_or_component_errors() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert!(f.get_by_node(NodeId(999), 0).is_err());
        assert_eq!(f.component_index("missing"), None);
    }

    #[test]
    fn protects_nodes_from_gc_after_node_and_submesh_drop() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let nid = n.id();
        let sm_handle = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[nid]).unwrap();
            insert(sm)
        };
        let field = NodeField::from_poi1(&sm_handle, vec!["T".into()]).unwrap();
        // Drop the Node and the SubMesh handle; the field still holds an incref.
        drop(n);
        drop(sm_handle);
        with_mut(&cfg, |c| assert_eq!(c.gc(), 0)).unwrap();
        with(&cfg, |c| assert!(c.is_alive(nid))).unwrap();
        drop(field);
        with_mut(&cfg, |c| assert_eq!(c.gc(), 1)).unwrap();
        with(&cfg, |c| assert!(!c.is_alive(nid))).unwrap();
    }

    #[test]
    fn snapshot_is_independent_of_later_submesh_growth() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let sm_handle = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            insert(sm)
        };
        let field = NodeField::from_poi1(&sm_handle, vec!["T".into()]).unwrap();
        assert_eq!(field.node_count(), 1);

        // Add another node + cell to the original SubMesh; field is unaffected.
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        with_mut(&sm_handle, |sm| sm.add_cell(&[b.id()])).unwrap().unwrap();
        assert_eq!(field.node_count(), 1);
    }

    #[test]
    fn to_poi1_submesh_mirrors_support() {
        let (cfg, nodes, sm) = make_poi1_with(3);
        let f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        // refcount before: Node + SubMesh + NodeField = 3 each
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 3)).unwrap();

        let sm2 = f.to_poi1_submesh().unwrap();
        // new submesh adds one more incref each
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 4)).unwrap();

        assert_eq!(sm2.cell_count(), 3);
        let conn = sm2.connectivity();
        for (i, n) in nodes.iter().enumerate() {
            assert_eq!(conn[i], n.id());
        }

        drop(sm2);
        // back to 3
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 3)).unwrap();
    }

    #[test]
    fn debug_and_display() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let f = NodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        let d = format!("{:?}", f);
        assert!(d.contains("NodeField"));
        assert!(d.contains("UX"));
        let s = format!("{}", f);
        assert!(s.contains("NodeField"));
        assert!(s.contains("2 node(s)"));
        assert!(s.contains("2 component(s)"));
        assert!(s.contains("UX, UY"));
    }

    // ── Index ────────────────────────────────────────────────────────────────

    #[test]
    fn index_read_write() {
        let (_cfg, nodes, sm) = make_poi1_with(2);
        let mut f = NodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        f[(nodes[0].id(), "UX")] = 7.0;
        f[(nodes[1].id(), "UY")] = -3.0;
        assert_eq!(f[(nodes[0].id(), "UX")], 7.0);
        assert_eq!(f[(nodes[1].id(), "UY")], -3.0);
        assert_eq!(f[(nodes[0].id(), "UY")], 0.0);
    }

    #[test]
    #[should_panic]
    fn index_unknown_node_panics() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let _ = f[(NodeId(999), "T")];
    }

    #[test]
    #[should_panic]
    fn index_unknown_component_panics() {
        let (_cfg, nodes, sm) = make_poi1_with(1);
        let f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let _ = f[(nodes[0].id(), "X")];
    }

    // ── Accès idiomatique ────────────────────────────────────────────────────

    #[test]
    fn value_and_set_value() {
        let (_cfg, nodes, sm) = make_poi1_with(2);
        let mut f = NodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        f.set_value(nodes[0].id(), "UY", 3.5).unwrap();
        assert_eq!(f.value(nodes[0].id(), "UY").unwrap(), 3.5);
        assert_eq!(f.value(nodes[1].id(), "UX").unwrap(), 0.0);
        assert!(f.value(NodeId(999), "UX").is_err());
        assert!(f.value(nodes[0].id(), "UZ").is_err());
        assert!(f.set_value(NodeId(999), "UX", 1.0).is_err());
        assert!(f.set_value(nodes[0].id(), "UZ", 1.0).is_err());
    }

    // ── add_fields ───────────────────────────────────────────────────────────

    #[test]
    fn add_fields_same_support() {
        let (_cfg, _nodes, sm) = make_poi1_with(3);
        let mut a = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let mut b = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        a.set(0, 0, 1.0).unwrap();
        b.set(0, 0, 2.0).unwrap();
        b.set(1, 0, 5.0).unwrap();
        let c = a.add_fields(&b).unwrap();
        assert_eq!(c.value(a.nodes[0], "T").unwrap(), 3.0);
        assert_eq!(c.value(a.nodes[1], "T").unwrap(), 5.0);
        assert_eq!(c.value(a.nodes[2], "T").unwrap(), 0.0);
    }

    #[test]
    fn add_fields_disjoint_components() {
        let (_cfg, nodes, sm) = make_poi1_with(2);
        let mut a = NodeField::from_poi1(&sm, vec!["UX".into()]).unwrap();
        let mut b = NodeField::from_poi1(&sm, vec!["UY".into()]).unwrap();
        a.set(0, 0, 10.0).unwrap();
        b.set(0, 0, 20.0).unwrap();
        let c = a.add_fields(&b).unwrap();
        assert_eq!(c.components(), &["UX", "UY"]);
        assert_eq!(c.value(nodes[0].id(), "UX").unwrap(), 10.0);
        assert_eq!(c.value(nodes[0].id(), "UY").unwrap(), 20.0);
    }

    #[test]
    fn add_fields_disjoint_nodes() {
        let cfg = insert(Configuration::new(1).unwrap());
        let na = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let nb = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let sm_a = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[na.id()]).unwrap();
            insert(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[nb.id()]).unwrap();
            insert(sm)
        };
        let mut a = NodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        let mut b = NodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        a.set(0, 0, 3.0).unwrap();
        b.set(0, 0, 7.0).unwrap();
        let c = a.add_fields(&b).unwrap();
        assert_eq!(c.node_count(), 2);
        assert_eq!(c.value(na.id(), "T").unwrap(), 3.0);
        assert_eq!(c.value(nb.id(), "T").unwrap(), 7.0);
    }

    #[test]
    fn add_fields_incompatible_cfg_errors() {
        let cfg1 = insert(Configuration::new(1).unwrap());
        let cfg2 = insert(Configuration::new(1).unwrap());
        let mk = |cfg: &Handle<Configuration>| {
            let n = Node::create_in(cfg.clone(), &[0.0]).unwrap();
            let sm = {
                let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
                sm.add_cell(&[n.id()]).unwrap();
                insert(sm)
            };
            NodeField::from_poi1(&sm, vec!["T".into()]).unwrap()
        };
        let a = mk(&cfg1);
        let b = mk(&cfg2);
        assert!(a.add_fields(&b).is_err());
    }

    // ── merge_fields ─────────────────────────────────────────────────────────

    #[test]
    fn merge_fields_compatible() {
        // a: nodes [na, nb] with T = [5.0, 3.0]
        // b: nodes [nb, nc] with T = [3.0, 9.0]  (nb shared, same value → compatible)
        let cfg = insert(Configuration::new(1).unwrap());
        let na = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let nb = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let nc = Node::create_in(cfg.clone(), &[2.0]).unwrap();
        let sm_a = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[na.id()]).unwrap();
            sm.add_cell(&[nb.id()]).unwrap();
            insert(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[nb.id()]).unwrap();
            sm.add_cell(&[nc.id()]).unwrap();
            insert(sm)
        };
        let mut a = NodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        let mut b = NodeField::from_poi1(&sm_b, vec!["T".into()]).unwrap();
        a.set(0, 0, 5.0).unwrap(); // na → 5.0
        a.set(1, 0, 3.0).unwrap(); // nb → 3.0
        b.set(0, 0, 3.0).unwrap(); // nb → 3.0 (same value, compatible)
        b.set(1, 0, 9.0).unwrap(); // nc → 9.0
        let c = a.merge_fields(&b).unwrap();
        assert_eq!(c.node_count(), 3);
        assert_eq!(c.value(na.id(), "T").unwrap(), 5.0);
        assert_eq!(c.value(nb.id(), "T").unwrap(), 3.0);
        assert_eq!(c.value(nc.id(), "T").unwrap(), 9.0);
    }

    #[test]
    fn merge_fields_conflict_errors() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let mut a = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let mut b = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        a.set(0, 0, 1.0).unwrap();
        b.set(0, 0, 2.0).unwrap();
        assert!(a.merge_fields(&b).is_err());
    }

    // ── Scalaires sur composante ──────────────────────────────────────────────

    #[test]
    fn component_scalar_ops() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let mut f = NodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        f.set(0, 0, 10.0).unwrap();
        f.set(1, 0, 20.0).unwrap();
        f.add_to_component("UX", 5.0).unwrap();
        assert_eq!(f.get(0, 0).unwrap(), 15.0);
        assert_eq!(f.get(1, 0).unwrap(), 25.0);
        assert_eq!(f.get(0, 1).unwrap(), 0.0); // UY unchanged
        f.sub_to_component("UX", 3.0).unwrap();
        assert_eq!(f.get(0, 0).unwrap(), 12.0);
        f.mul_to_component("UX", 2.0).unwrap();
        assert_eq!(f.get(0, 0).unwrap(), 24.0);
        f.div_to_component("UX", 4.0).unwrap();
        assert_eq!(f.get(0, 0).unwrap(), 6.0);
    }

    #[test]
    fn div_to_component_zero_errors() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let mut f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert!(f.div_to_component("T", 0.0).is_err());
    }

    #[test]
    fn component_scalar_unknown_component_errors() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let mut f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert!(f.add_to_component("X", 1.0).is_err());
        assert!(f.sub_to_component("X", 1.0).is_err());
        assert!(f.mul_to_component("X", 1.0).is_err());
        assert!(f.div_to_component("X", 1.0).is_err());
    }

    // ── Clone ────────────────────────────────────────────────────────────────

    #[test]
    fn clone_is_independent() {
        let (cfg, nodes, sm) = make_poi1_with(2);
        let mut f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set(0, 0, 42.0).unwrap();
        let g = f.clone();
        // Clone holds extra increfs
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 4)).unwrap();
        // Mutation of f does not affect g
        f.set(0, 0, 99.0).unwrap();
        assert_eq!(g.get(0, 0).unwrap(), 42.0);
        drop(g);
        // Back to 3 after clone is dropped
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 3)).unwrap();
    }

    // ── Opérateurs +,-,*,/ avec f64 ─────────────────────────────────────────

    #[test]
    fn operator_add_f64() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let mut f = NodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        f.set(0, 0, 1.0).unwrap();
        f.set(1, 1, 3.0).unwrap();
        let g = &f + 10.0;
        assert_eq!(g.get(0, 0).unwrap(), 11.0);
        assert_eq!(g.get(1, 1).unwrap(), 13.0);
        assert_eq!(g.get(0, 1).unwrap(), 10.0); // 0.0 + 10.0
        // f is still usable (reference version was used)
        assert_eq!(f.get(0, 0).unwrap(), 1.0);
    }

    #[test]
    fn operator_sub_mul_div_f64() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let mut f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set(0, 0, 12.0).unwrap();
        assert_eq!((f.clone() - 2.0).get(0, 0).unwrap(), 10.0);
        assert_eq!((f.clone() * 3.0).get(0, 0).unwrap(), 36.0);
        assert_eq!((f * 4.0 / 2.0).get(0, 0).unwrap(), 24.0);
    }

    // ── restrict ─────────────────────────────────────────────────────────────

    #[test]
    fn restrict_subset() {
        use crate::mesh::Mesh;
        let (cfg, nodes, _sm) = make_poi1_with(3);
        // Build a full field on all 3 nodes
        let sm_all = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            for n in &nodes {
                sm.add_cell(&[n.id()]).unwrap();
            }
            insert(sm)
        };
        let mut f = NodeField::from_poi1(&sm_all, vec!["T".into(), "P".into()]).unwrap();
        f.set(0, 0, 1.0).unwrap();
        f.set(1, 0, 2.0).unwrap();
        f.set(2, 0, 3.0).unwrap();

        // Build a mesh with only nodes[0] and nodes[2]
        let mut m = Mesh::with_element_type(cfg.clone(), ElementType::POI1);
        m.add_cell(&[nodes[0].id()]).unwrap();
        m.add_cell(&[nodes[2].id()]).unwrap();

        let r = f.restrict(&m).unwrap();
        assert_eq!(r.node_count(), 2);
        assert_eq!(r.components(), &["T", "P"]);
        assert_eq!(r.value(nodes[0].id(), "T").unwrap(), 1.0);
        assert_eq!(r.value(nodes[2].id(), "T").unwrap(), 3.0);
        assert_eq!(r.value(nodes[0].id(), "P").unwrap(), 0.0); // absent → 0
    }

    #[test]
    fn restrict_node_absent_from_field_gives_zero() {
        use crate::mesh::Mesh;
        let cfg = insert(Configuration::new(1).unwrap());
        let na = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let nb = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let sm_a = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[na.id()]).unwrap();
            insert(sm)
        };
        let mut f = NodeField::from_poi1(&sm_a, vec!["T".into()]).unwrap();
        f.set(0, 0, 7.0).unwrap();

        // Mesh contains nb which is NOT in the field
        let mut m = Mesh::with_element_type(cfg.clone(), ElementType::POI1);
        m.add_cell(&[na.id()]).unwrap();
        m.add_cell(&[nb.id()]).unwrap();

        let r = f.restrict(&m).unwrap();
        assert_eq!(r.node_count(), 2);
        assert_eq!(r.value(na.id(), "T").unwrap(), 7.0);
        assert_eq!(r.value(nb.id(), "T").unwrap(), 0.0);
    }

    #[test]
    fn restrict_incompatible_cfg_errors() {
        use crate::mesh::Mesh;
        let cfg1 = insert(Configuration::new(1).unwrap());
        let cfg2 = insert(Configuration::new(1).unwrap());
        let n1 = Node::create_in(cfg1.clone(), &[0.0]).unwrap();
        let sm1 = {
            let mut sm = SubMesh::new(cfg1.clone(), ElementType::POI1);
            sm.add_cell(&[n1.id()]).unwrap();
            insert(sm)
        };
        let f = NodeField::from_poi1(&sm1, vec!["T".into()]).unwrap();
        let m2 = Mesh::new(cfg2);
        assert!(f.restrict(&m2).is_err());
    }
}
