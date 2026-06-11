//! NodeField — multi-component values on a POI1 submesh.
//!
//! A [`NodeField`] stores one or more named components per node of a
//! support defined by a POI1 [`SubMesh`] (a list of nodes). The field
//! holds a `Handle<SubMesh>` on its support: the SubMesh is the
//! single owner of per-node refcounts in the [`Configuration`] (its
//! `add_cell` increfs, its `Drop` decrefs). The field itself does no
//! per-node refcount bookkeeping — keeping a clone of the support
//! handle is enough to keep the SubMesh (and therefore its nodes)
//! alive.
//!
//! By project convention, a SubMesh's connectivity is frozen after
//! creation (see project memory), so the field caches the node list
//! once at construction for fast lookup.
//!
//! The default value of every component is `0.0`.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::Configuration;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::containers::node_field::NodeField;
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

use crate::aggregate::Aggregate;
use crate::containers::mesh::{Configuration, NodeId};
use crate::containers::mesh::ElementType;
use crate::error::{PyrucastError, Result};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::store::{insert, with, Handle};
#[cfg(test)]
use crate::store::with_mut;
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
    /// POI1 SubMesh owning the per-node refcounts. The field keeps a
    /// clone of this handle for its whole lifetime; that is the only
    /// thing keeping the support (and its nodes) alive.
    support: Handle<SubMesh>,
    /// Cached connectivity of `support` (POI1 ⇒ one node per cell).
    /// Frozen at construction by [[project-submesh-immutable-size]].
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

        let nodes: Vec<NodeId> = with(submesh, |sm| -> Result<_> {
            if sm.element_type() != ElementType::POI1 {
                return Err(PyrucastError::Message(format!(
                    "NodeField requires a POI1 SubMesh, got {}",
                    sm.element_type()
                )));
            }
            // POI1: connectivity is exactly the node list (1 node per cell).
            Ok(sm.connectivity().to_vec())
        })??;

        let n_nodes = nodes.len();
        let n_comp = components.len();
        Ok(NodeField {
            support: submesh.clone(),
            nodes,
            components,
            values: vec![0.0; n_nodes * n_comp],
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

    /// Node ids this field is defined on, in support order.
    pub(crate) fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// Handle to the owning `Configuration` (derived from the support).
    pub fn configuration(&self) -> Handle<Configuration> {
        with(&self.support, |sm| sm.configuration())
            .expect("NodeField support handle is held by self → must be alive")
    }

    /// Handle to the POI1 SubMesh backing this field's support.
    pub fn support(&self) -> Handle<SubMesh> {
        self.support.clone()
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
    /// use pyrucast::aggregate::Aggregate;
    /// use pyrucast::containers::mesh::Configuration;
    /// use pyrucast::containers::mesh::ElementType;
    /// use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// use pyrucast::containers::mesh::Node;
    /// use pyrucast::containers::node_field::NodeField;
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
    /// let sm2 = field.support_submesh().unwrap();
    /// assert_eq!(sm2.cell_count(), 2);
    /// // Verify node order via Mesh::node (public API).
    /// let mut m = Mesh::empty();
    /// m.add_sub(insert(sm2)).unwrap();
    /// assert_eq!(m.node(0, 0, 0).unwrap().id(), a.id());
    /// assert_eq!(m.node(0, 1, 0).unwrap().id(), b.id());
    /// ```
    pub fn support_submesh(&self) -> Result<SubMesh> {
        SubMesh::poi1_from_node_ids(self.configuration(), &self.nodes)
    }

    /// Build a [`Mesh`] with a single POI1 submesh mirroring the support of
    /// this field.
    pub fn support_mesh(&self) -> Result<Mesh> {
        let sm_handle = insert(self.support_submesh()?);
        let mut mesh = Mesh::empty();
        mesh.add_sub(sm_handle)?;
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
    /// Materialises a fresh POI1 SubMesh holding the per-node refcounts:
    /// if any `add_cell` fails, the partial SubMesh's `Drop` rolls back
    /// the increfs already done.
    pub(crate) fn new_with_nodes(
        cfg: Handle<Configuration>,
        nodes: Vec<NodeId>,
        components: Vec<String>,
    ) -> Result<Self> {
        let n_nodes = nodes.len();
        let n_comp = components.len();
        let sm = SubMesh::poi1_from_node_ids(cfg, &nodes)?;
        Ok(NodeField {
            support: insert(sm),
            nodes,
            components,
            values: vec![0.0; n_nodes * n_comp],
        })
    }

    // ── Helpers privés ──────────────────────────────────────────────────────

    pub(crate) fn component_value_opt(&self, nid: NodeId, comp: &str) -> Option<f64> {
        let ni = self.index_of(nid)?;
        let ci = self.component_index(comp)?;
        Some(self.values[ni * self.components.len() + ci])
    }

    fn get_or_default(&self, nid: NodeId, comp: &str) -> f64 {
        self.component_value_opt(nid, comp).unwrap_or(0.0)
    }

    pub(crate) fn check_compatible(&self, other: &NodeField) -> Result<()> {
        let a = self.configuration();
        let b = other.configuration();
        if a.index() != b.index() || a.generation() != b.generation() {
            return Err(PyrucastError::Message(
                "fields are not attached to the same Configuration".into(),
            ));
        }
        Ok(())
    }

    /// Returns (union_components, union_nodes): self's items first, then other's extras.
    pub(crate) fn union_layout(&self, other: &NodeField) -> (Vec<String>, Vec<NodeId>) {
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

impl crate::containers::field::SubField for NodeField {
    fn components(&self) -> &[String] {
        &self.components
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
}

impl fmt::Debug for NodeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Bounded structure only — the per-node values live in `dump()`.
        f.debug_struct("NodeField")
            .field("support", &self.support)
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

impl crate::dump::Dump for NodeField {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        use crate::dump::{fmt_float, table};
        let ncomp = self.components.len();
        let mut headers = Vec::with_capacity(ncomp + 1);
        headers.push("node".to_string());
        headers.extend(self.components.iter().cloned());
        let rows: Vec<Vec<String>> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, nid)| {
                let mut row = Vec::with_capacity(ncomp + 1);
                row.push(nid.to_string());
                for c in 0..ncomp {
                    row.push(fmt_float(self.values[i * ncomp + c], opts.precision));
                }
                row
            })
            .collect();
        format!("{self}\n{}", table(&headers, &rows, opts))
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
        // Cloning the support Handle bumps the SubMesh's store refcount;
        // per-node refcounts in the Configuration are already covered by
        // the shared SubMesh.
        NodeField {
            support: self.support.clone(),
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

/// Component-wise addition of two fields (`&a + &b`).
///
/// Missing nodes or components in either field are treated as `0.0`. Both
/// fields must be attached to the same `Configuration`. The result's
/// component list is the union of both fields' (self first), and similarly
/// for nodes. Returns `Result` because the configuration check can fail —
/// same convention as `&mesh1 + &mesh2`.
impl Add<&NodeField> for &NodeField {
    type Output = Result<NodeField>;
    fn add(self, rhs: &NodeField) -> Result<NodeField> {
        self.check_compatible(rhs)?;
        let (components, nodes) = self.union_layout(rhs);
        let mut result =
            NodeField::new_with_nodes(self.configuration(), nodes.clone(), components.clone())?;
        let ncomp = components.len();
        for (ni, &nid) in nodes.iter().enumerate() {
            for (ci, comp) in components.iter().enumerate() {
                result.values[ni * ncomp + ci] =
                    self.get_or_default(nid, comp) + rhs.get_or_default(nid, comp);
            }
        }
        Ok(result)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::Node;
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
    fn shares_support_refcounts() {
        let (cfg, nodes, sm) = make_poi1_with(2);
        // Each node has refcount = 2 (Node + SubMesh).
        with(&cfg, |c| {
            assert_eq!(c.refcount(nodes[0].id()), 2);
            assert_eq!(c.refcount(nodes[1].id()), 2);
        })
        .unwrap();
        let f = NodeField::from_poi1(&sm, vec!["P".into()]).unwrap();
        // The field shares the SubMesh handle, so per-node refcounts
        // are unchanged.
        with(&cfg, |c| {
            assert_eq!(c.refcount(nodes[0].id()), 2);
            assert_eq!(c.refcount(nodes[1].id()), 2);
        })
        .unwrap();
        drop(f);
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
        // Drop the Node and the user's SubMesh handle; the field still
        // holds a clone of the SubMesh handle, which keeps the SubMesh
        // alive, which keeps the nodes alive.
        drop(n);
        drop(sm_handle);
        with_mut(&cfg, |c| assert_eq!(c.gc(), 0)).unwrap();
        with(&cfg, |c| assert!(c.is_alive(nid))).unwrap();
        drop(field);
        with_mut(&cfg, |c| assert_eq!(c.gc(), 1)).unwrap();
        with(&cfg, |c| assert!(!c.is_alive(nid))).unwrap();
    }

    #[test]
    fn support_submesh_mirrors_field_nodes() {
        let (cfg, nodes, sm) = make_poi1_with(3);
        let f = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        // refcount before: Node + SubMesh = 2 each (the field shares
        // the user's SubMesh, no extra per-node incref).
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 2)).unwrap();

        let sm2 = f.support_submesh().unwrap();
        // the freshly built SubMesh adds one incref each → 3
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 3)).unwrap();

        assert_eq!(sm2.cell_count(), 3);
        let conn = sm2.connectivity();
        for (i, n) in nodes.iter().enumerate() {
            assert_eq!(conn[i], n.id());
        }

        drop(sm2);
        // back to 2
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 2)).unwrap();
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

    #[test]
    fn dump_renders_value_table_and_debug_is_bounded() {
        use crate::dump::{Dump, DumpOptions};
        let (_cfg, nodes, sm) = make_poi1_with(2);
        let mut f = NodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        f.set_value(nodes[0].id(), "UX", 1.25).unwrap();

        let dumped = f.render(&DumpOptions::default());
        assert!(dumped.contains("UX") && dumped.contains("UY"), "headers:\n{dumped}");
        assert!(dumped.contains("1.250"), "value at default precision:\n{dumped}");
        assert_eq!(dumped.lines().count(), 4, "summary + header + 2 rows:\n{dumped}");

        // Debug must stay bounded: structure, never the value buffer.
        let dbg = format!("{f:?}");
        assert!(dbg.contains("node_count"), "{dbg}");
        assert!(!dbg.contains("1.25"), "Debug must not leak values: {dbg}");
    }

    // ── &a + &b (field addition) ───────────────────────────────────────────

    #[test]
    fn add_fields_same_support() {
        let (_cfg, _nodes, sm) = make_poi1_with(3);
        let mut a = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let mut b = NodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        a.set(0, 0, 1.0).unwrap();
        b.set(0, 0, 2.0).unwrap();
        b.set(1, 0, 5.0).unwrap();
        let c = (&a + &b).unwrap();
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
        let c = (&a + &b).unwrap();
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
        let c = (&a + &b).unwrap();
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
        assert!((&a + &b).is_err());
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
        // Both fields share the same SubMesh handle, so per-node
        // refcounts in the Configuration stay at 2 (Node + SubMesh).
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 2)).unwrap();
        // Mutation of f does not affect g (values are independent).
        f.set(0, 0, 99.0).unwrap();
        assert_eq!(g.get(0, 0).unwrap(), 42.0);
        drop(g);
        with(&cfg, |c| assert_eq!(c.refcount(nodes[0].id()), 2)).unwrap();
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

}
