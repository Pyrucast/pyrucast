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
}
