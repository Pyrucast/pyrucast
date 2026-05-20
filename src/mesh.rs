//! Mesh — collection of homogeneous submeshes (one element type per
//! submesh).
//!
//! Hierarchy:
//!
//! - [`SubMesh`] — every cell of a single [`ElementType`]. Stores the
//!   connectivity flat (`Vec<NodeId>`, length `cell_count * nodes_per_cell`).
//!   RAII referencing: `add_cell` increments the node refcounts in the
//!   `Configuration`; the `SubMesh`'s `Drop` decrements every referenced
//!   node.
//! - [`Mesh`] — aggregate of SubMeshes attached to the same `Configuration`.
//!
//! The POI1 case is deliberately degenerate: a POI1 submesh is exactly a
//! list of nodes.
//!
//! # Example
//!
//! ```
//! use pyrucast::configuration::Configuration;
//! use pyrucast::element_type::ElementType;
//! use pyrucast::mesh::SubMesh;
//! use pyrucast::node::Node;
//! use pyrucast::store::{insert, with, with_mut};
//!
//! let cfg = insert(Configuration::new(2).unwrap());
//! let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
//! let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();
//!
//! let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
//! sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//! assert_eq!(sm.cell_count(), 1);
//!
//! // The SubMesh holds refs on the 3 nodes, in addition to the `Node`s.
//! with(&cfg, |c| assert_eq!(c.refcount(a.id()), 2)).unwrap();
//! drop(sm);  // decrements the referenced nodes
//! with(&cfg, |c| assert_eq!(c.refcount(a.id()), 1)).unwrap();
//! ```

use crate::configuration::{Configuration, NodeId};
use crate::element_type::ElementType;
use crate::error::{PyrucastError, Result};
use crate::node::Node;
use crate::store::{insert, with, with_mut, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── SubMesh ────────────────────────────────────────────────────────────────

/// Submesh: every cell of a single [`ElementType`].
///
/// The connectivity is stored flat; each cell occupies
/// `element_type.nodes_per_cell()` contiguous entries.
#[derive(Serialize, Deserialize)]
pub struct SubMesh {
    element_type: ElementType,
    config: Handle<Configuration>,
    /// Flat connectivity: cell `i` occupies `[i*npc, (i+1)*npc)`.
    connectivity: Vec<NodeId>,
}

impl SubMesh {
    /// Create an empty submesh for the given element type, attached to `config`.
    pub fn new(config: Handle<Configuration>, element_type: ElementType) -> Self {
        Self {
            element_type,
            config,
            connectivity: Vec::new(),
        }
    }

    /// Add a cell. The length of `nodes` must equal
    /// `element_type.nodes_per_cell()`, and each node must be alive in the
    /// `Configuration`; each node is increfed. On increment failure
    /// (invalid / collected id), the increfs already performed for this
    /// cell are rolled back.
    pub fn add_cell(&mut self, nodes: &[NodeId]) -> Result<usize> {
        let npc = self.element_type.nodes_per_cell();
        if nodes.len() != npc {
            return Err(PyrucastError::Message(format!(
                "add_cell({}): expected {} nodes, got {}",
                self.element_type,
                npc,
                nodes.len()
            )));
        }
        let result: Result<()> = with_mut(&self.config, |c| {
            let mut acquired = 0usize;
            for &n in nodes {
                if let Err(e) = c.incref(n) {
                    // Roll back the increfs already done for this cell.
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
        let idx = self.connectivity.len() / npc;
        self.connectivity.extend_from_slice(nodes);
        Ok(idx)
    }

    /// Element type of the submesh.
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Number of cells in the submesh.
    pub fn cell_count(&self) -> usize {
        self.connectivity.len() / self.element_type.nodes_per_cell()
    }

    /// Flat connectivity buffer (all cells concatenated).
    pub(crate) fn connectivity(&self) -> &[NodeId] {
        &self.connectivity
    }

    /// Handle to the owning `Configuration` (internal clone).
    pub fn configuration(&self) -> Handle<Configuration> {
        self.config.clone()
    }
}

impl Drop for SubMesh {
    fn drop(&mut self) {
        // One lock acquisition for all decrefs.
        let _ = with_mut(&self.config, |c| {
            for &n in &self.connectivity {
                let _ = c.decref(n);
            }
        });
    }
}

impl fmt::Debug for SubMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubMesh")
            .field("element_type", &self.element_type)
            .field("cell_count", &self.cell_count())
            .finish()
    }
}

impl fmt::Display for SubMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SubMesh<{}>: {} cell(s)",
            self.element_type,
            self.cell_count()
        )
    }
}

// ─── Mesh ───────────────────────────────────────────────────────────────────

/// Mesh: aggregate of submeshes attached to the same `Configuration`.
#[derive(Serialize, Deserialize)]
pub struct Mesh {
    config: Handle<Configuration>,
    submeshes: Vec<Handle<SubMesh>>,
}

impl Mesh {
    /// Create an empty mesh attached to `config`.
    pub fn new(config: Handle<Configuration>) -> Self {
        Self {
            config,
            submeshes: Vec::new(),
        }
    }

    /// Add a submesh. Requires that the submesh's `Configuration` matches
    /// the mesh's.
    pub fn add_submesh(&mut self, sm: Handle<SubMesh>) -> Result<()> {
        let sm_cfg = with(&sm, |s| s.configuration())?;
        if sm_cfg.index() != self.config.index() || sm_cfg.generation() != self.config.generation()
        {
            return Err(PyrucastError::Message(
                "add_submesh: submesh attached to a different Configuration".into(),
            ));
        }
        self.submeshes.push(sm);
        Ok(())
    }

    /// Number of submeshes.
    pub fn submesh_count(&self) -> usize {
        self.submeshes.len()
    }

    /// Return a clone of the handle to the submesh at index `idx`.
    pub fn submesh(&self, idx: usize) -> Result<Handle<SubMesh>> {
        self.submeshes
            .get(idx)
            .cloned()
            .ok_or_else(|| PyrucastError::Message(format!("submesh: index {} out of bounds", idx)))
    }

    /// Total cells in the mesh (sum across submeshes).
    pub fn cell_count(&self) -> Result<usize> {
        let mut total = 0usize;
        for sm in &self.submeshes {
            total += with(sm, |s| s.cell_count())?;
        }
        Ok(total)
    }

    /// Handle to the `Configuration` (internal clone).
    pub fn configuration(&self) -> Handle<Configuration> {
        self.config.clone()
    }

    /// Create a mesh pre-loaded with one empty submesh of `element_type`.
    pub fn with_element_type(config: Handle<Configuration>, element_type: ElementType) -> Self {
        let sm = insert(SubMesh::new(config.clone(), element_type));
        let mut mesh = Self {
            config,
            submeshes: Vec::new(),
        };
        mesh.submeshes.push(sm);
        mesh
    }

    /// Add a cell directly when the mesh has exactly one submesh.
    pub fn add_cell(&mut self, nodes: &[NodeId]) -> Result<usize> {
        if self.submeshes.len() != 1 {
            return Err(PyrucastError::Message(
                "add_cell: mesh must have exactly one submesh".into(),
            ));
        }
        with_mut(&self.submeshes[0], |s| s.add_cell(nodes))?
    }

    /// Element type of each submesh, in order.
    pub fn element_types(&self) -> Result<Vec<ElementType>> {
        self.submeshes
            .iter()
            .map(|sm| with(sm, |s| s.element_type()))
            .collect()
    }

    /// Cell count of each submesh, in order.
    pub fn cell_counts(&self) -> Result<Vec<usize>> {
        self.submeshes
            .iter()
            .map(|sm| with(sm, |s| s.cell_count()))
            .collect()
    }

    /// Node at position `node_idx` in cell `cell_idx` of submesh `submesh_idx`.
    pub fn node(&self, submesh_idx: usize, cell_idx: usize, node_idx: usize) -> Result<Node> {
        let sm = self.submesh(submesh_idx)?;
        let nid: NodeId = with(&sm, |s| {
            let npc = s.element_type.nodes_per_cell();
            let n = s.cell_count();
            if cell_idx >= n {
                return Err(PyrucastError::Message(format!(
                    "node: cell index {} ≥ cell_count {}",
                    cell_idx, n
                )));
            }
            s.connectivity()
                .get(cell_idx * npc + node_idx)
                .copied()
                .ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "node: node index {} ≥ nodes_per_cell {}",
                        node_idx, npc
                    ))
                })
        })??;
        Node::acquire(self.config.clone(), nid)
    }

    /// Create a POI1 mesh containing all live nodes of `config`.
    pub fn from_live_nodes(config: Handle<Configuration>) -> Result<Mesh> {
        let node_ids: Vec<NodeId> = with(&config, |c| c.iter_live().collect())?;
        let mut mesh = Mesh::with_element_type(config, ElementType::POI1);
        for nid in node_ids {
            mesh.add_cell(&[nid])?;
        }
        Ok(mesh)
    }

    /// Return a new mesh containing all submeshes of `self` followed by all
    /// submeshes of `other`. Both meshes must share the same `Configuration`.
    pub fn merge(&self, other: &Mesh) -> Result<Mesh> {
        if self.config.index() != other.config.index()
            || self.config.generation() != other.config.generation()
        {
            return Err(PyrucastError::Message(
                "merge: meshes are attached to different Configurations".into(),
            ));
        }
        let mut result = Mesh::new(self.config.clone());
        for sm in self.submeshes.iter().chain(other.submeshes.iter()) {
            result.submeshes.push(sm.clone());
        }
        Ok(result)
    }
}

impl std::ops::Add<&Mesh> for &Mesh {
    type Output = Result<Mesh>;
    fn add(self, rhs: &Mesh) -> Result<Mesh> {
        self.merge(rhs)
    }
}

impl fmt::Debug for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mesh")
            .field("submesh_count", &self.submeshes.len())
            .finish()
    }
}

impl fmt::Display for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = self.cell_count().unwrap_or(0);
        write!(
            f,
            "Mesh: {} submesh(es), {} cell(s) total",
            self.submeshes.len(),
            total
        )
    }
}

// ─── Python binding ─────────────────────────────────────────────────────────

#[cfg(feature = "extension-module")]
mod python {
    use super::*;
    use crate::configuration::PyConfiguration;
    use crate::node::PyNode;
    use crate::store::insert;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// Python wrapper for [`SubMesh`].
    #[pyclass(name = "SubMesh")]
    pub struct PySubMesh {
        pub(crate) handle: Handle<SubMesh>,
    }

    #[pymethods]
    impl PySubMesh {
        #[new]
        fn py_new(config: PyRef<PyConfiguration>, element_type: &str) -> PyResult<Self> {
            let et = ElementType::from_name(element_type).ok_or_else(|| {
                PyValueError::new_err(format!("unknown element type: {element_type}"))
            })?;
            let cfg_handle = config.handle.clone();
            let sm = SubMesh::new(cfg_handle, et);
            Ok(Self { handle: insert(sm) })
        }

        #[getter]
        fn element_type(&self) -> PyResult<String> {
            Ok(with(&self.handle, |s| s.element_type().name().to_string())?)
        }

        fn add_cell(&self, nodes: Vec<u32>) -> PyResult<usize> {
            let nodes_typed: Vec<NodeId> = nodes.iter().map(|&i| NodeId(i)).collect();
            let idx = with_mut(&self.handle, move |s| s.add_cell(&nodes_typed))??;
            Ok(idx)
        }

        fn cell_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |s| s.cell_count())?)
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |s| format!("{:?}", s))?)
        }

        fn __str__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |s| format!("{}", s))?)
        }
    }

    /// Python wrapper for [`Mesh`].
    #[pyclass(name = "Mesh")]
    pub struct PyMesh {
        pub(crate) handle: Handle<Mesh>,
    }

    #[pymethods]
    impl PyMesh {
        /// `Mesh(config)` — empty mesh.
        /// `Mesh(config, element_type)` — mesh with one pre-created submesh.
        #[new]
        #[pyo3(signature = (config, element_type=None))]
        fn py_new(config: PyRef<PyConfiguration>, element_type: Option<&str>) -> PyResult<Self> {
            let cfg = config.handle.clone();
            let mesh = match element_type {
                Some(et_str) => {
                    let et = ElementType::from_name(et_str).ok_or_else(|| {
                        PyValueError::new_err(format!("unknown element type: {et_str}"))
                    })?;
                    Mesh::with_element_type(cfg, et)
                }
                None => Mesh::new(cfg),
            };
            Ok(Self { handle: insert(mesh) })
        }

        fn add_submesh(&self, sm: PyRef<PySubMesh>) -> PyResult<()> {
            let sm_handle = sm.handle.clone();
            with_mut(&self.handle, |m| m.add_submesh(sm_handle))??;
            Ok(())
        }

        fn add_cell(&self, nodes: Vec<u32>) -> PyResult<usize> {
            let nodes_typed: Vec<NodeId> = nodes.iter().map(|&i| NodeId(i)).collect();
            let idx = with_mut(&self.handle, move |m| m.add_cell(&nodes_typed))??;
            Ok(idx)
        }

        #[getter]
        fn element_type(&self) -> PyResult<Option<String>> {
            let maybe_sm = with(&self.handle, |m| -> Option<Handle<SubMesh>> {
                if m.submesh_count() == 1 {
                    m.submesh(0).ok()
                } else {
                    None
                }
            })?;
            match maybe_sm {
                Some(h) => Ok(Some(with(&h, |sm| sm.element_type().name().to_string())?)),
                None => Ok(None),
            }
        }

        fn element_types(&self) -> PyResult<Vec<String>> {
            let types = with(&self.handle, |m| m.element_types())??;
            Ok(types.into_iter().map(|et| et.name().to_string()).collect())
        }

        fn cell_counts(&self) -> PyResult<Vec<usize>> {
            Ok(with(&self.handle, |m| m.cell_counts())??)
        }

        fn node(&self, submesh_idx: usize, cell_idx: usize, node_idx: usize) -> PyResult<PyNode> {
            let node = with(&self.handle, |m| m.node(submesh_idx, cell_idx, node_idx))??;
            Ok(PyNode::from_node(node))
        }

        #[classmethod]
        fn from_live_nodes(
            _cls: &pyo3::Bound<'_, pyo3::types::PyType>,
            config: PyRef<PyConfiguration>,
        ) -> PyResult<Self> {
            let mesh = Mesh::from_live_nodes(config.handle.clone())?;
            Ok(Self { handle: insert(mesh) })
        }

        fn __add__(&self, other: PyRef<PyMesh>) -> PyResult<PyMesh> {
            let other_handle = other.handle.clone();
            let mesh = with(&self.handle, |a| {
                with(&other_handle, |b| a.merge(b))
            })???;
            Ok(PyMesh { handle: insert(mesh) })
        }

        fn submesh_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |m| m.submesh_count())?)
        }

        fn cell_count(&self) -> PyResult<usize> {
            Ok(with(&self.handle, |m| m.cell_count())??)
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |m| format!("{:?}", m))?)
        }

        fn __str__(&self) -> PyResult<String> {
            Ok(with(&self.handle, |m| format!("{}", m))?)
        }
    }
}

#[cfg(feature = "extension-module")]
pub use python::{PyMesh, PySubMesh};

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::store::{insert, with};

    #[test]
    fn submesh_poi1_is_node_list() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        sm.add_cell(&[b.id()]).unwrap();
        assert_eq!(sm.cell_count(), 2);
        assert_eq!(sm.connectivity()[0], a.id());
        assert_eq!(sm.connectivity()[1], b.id());
    }

    #[test]
    fn submesh_tri3_increfs_and_drop_decrefs() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        // SubMesh increfed each of the 3 nodes, in addition to the Nodes.
        with(&cfg, |cf| {
            assert_eq!(cf.refcount(a.id()), 2);
            assert_eq!(cf.refcount(b.id()), 2);
            assert_eq!(cf.refcount(c.id()), 2);
        })
        .unwrap();
        drop(sm);
        with(&cfg, |cf| {
            assert_eq!(cf.refcount(a.id()), 1);
            assert_eq!(cf.refcount(b.id()), 1);
            assert_eq!(cf.refcount(c.id()), 1);
        })
        .unwrap();
    }

    #[test]
    fn submesh_add_cell_invalid_arity() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        let err = sm.add_cell(&[a.id()]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
        // No increment should have survived the failure.
        with(&cfg, |cf| assert_eq!(cf.refcount(a.id()), 1)).unwrap();
    }

    #[test]
    fn submesh_add_cell_collected_node_rollback() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let dead_id = with_mut(&cfg, |c| c.add_node(&[2.0])).unwrap().unwrap();
        // dead_id starts at refcount=1; decrement then collect.
        with_mut(&cfg, |c| {
            c.decref(dead_id).unwrap();
            assert_eq!(c.gc(), 1);
        })
        .unwrap();

        let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
        // a (live), b (live), dead_id (collected) → add_cell fails after
        // increfing a and b. The rollback must undo those increfs.
        let err = sm.add_cell(&[a.id(), b.id(), dead_id]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
        with(&cfg, |cf| {
            assert_eq!(cf.refcount(a.id()), 1, "a must be rolled back");
            assert_eq!(cf.refcount(b.id()), 1, "b must be rolled back");
        })
        .unwrap();
        assert_eq!(sm.cell_count(), 0);
    }

    #[test]
    fn mesh_aggregates_submeshes_same_config() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let cc = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let sm_pts = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm.add_cell(&[b.id()]).unwrap();
            insert(sm)
        };
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), cc.id()]).unwrap();
            insert(sm)
        };

        let mut mesh = Mesh::new(cfg.clone());
        mesh.add_submesh(sm_pts).unwrap();
        mesh.add_submesh(sm_tri).unwrap();
        assert_eq!(mesh.submesh_count(), 2);
        assert_eq!(mesh.cell_count().unwrap(), 3); // 2 points + 1 triangle
    }

    #[test]
    fn mesh_rejects_submesh_from_other_configuration() {
        let cfg1 = insert(Configuration::new(2).unwrap());
        let cfg2 = insert(Configuration::new(2).unwrap());

        let sm = insert(SubMesh::new(cfg1.clone(), ElementType::POI1));
        let mut mesh = Mesh::new(cfg2);
        let err = mesh.add_submesh(sm).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn mesh_element_types_and_cell_counts() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::with_element_type(cfg.clone(), ElementType::POI1);
        m.add_cell(&[a.id()]).unwrap();
        m.add_cell(&[b.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        m.add_submesh(sm_tri).unwrap();

        assert_eq!(
            m.element_types().unwrap(),
            vec![ElementType::POI1, ElementType::TRI3]
        );
        assert_eq!(m.cell_counts().unwrap(), vec![2, 1]);
    }

    #[test]
    fn mesh_node_access_by_indices() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::with_element_type(cfg.clone(), ElementType::TRI3);
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let n = m.node(0, 0, 0).unwrap();
        assert_eq!(n.id(), a.id()); // noeud 0 de l'élément 0 = a
        assert!(m.node(1, 0, 0).is_err()); // sous-maillage hors bornes
        assert!(m.node(0, 1, 0).is_err()); // cellule hors bornes
        assert!(m.node(0, 0, 3).is_err()); // noeud hors bornes (TRI3 : indices 0..2)
    }

    #[test]
    fn mesh_from_live_nodes() {
        let cfg = insert(Configuration::new(1).unwrap());
        let _a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let _b = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let _c = Node::create_in(cfg.clone(), &[2.0]).unwrap();

        let m = Mesh::from_live_nodes(cfg.clone()).unwrap();
        assert_eq!(m.element_types().unwrap(), vec![ElementType::POI1]);
        assert_eq!(m.cell_count().unwrap(), 3);

        // from_live_nodes est un snapshot : le maillage m tient les refs,
        // un second appel sur la même configuration donne le même résultat.
        let m2 = Mesh::from_live_nodes(cfg).unwrap();
        assert_eq!(m2.cell_count().unwrap(), 3);
    }

    #[test]
    fn mesh_merge_combines_submeshes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m1 = Mesh::with_element_type(cfg.clone(), ElementType::POI1);
        m1.add_cell(&[a.id()]).unwrap();
        m1.add_cell(&[b.id()]).unwrap();

        let mut m2 = Mesh::with_element_type(cfg.clone(), ElementType::TRI3);
        m2.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let merged = (&m1 + &m2).unwrap();
        assert_eq!(merged.submesh_count(), 2);
        assert_eq!(merged.cell_count().unwrap(), 3); // 2 POI1 + 1 TRI3
    }

    #[test]
    fn mesh_merge_rejects_different_configurations() {
        let cfg1 = insert(Configuration::new(2).unwrap());
        let cfg2 = insert(Configuration::new(2).unwrap());
        let err = Mesh::new(cfg1).merge(&Mesh::new(cfg2)).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn debug_and_display_submesh_and_mesh() {
        let cfg = insert(Configuration::new(1).unwrap());
        let sm = SubMesh::new(cfg.clone(), ElementType::SEG2);
        let d = format!("{:?}", sm);
        let s = format!("{}", sm);
        assert!(d.contains("SubMesh"));
        assert!(s.contains("SEG2"));

        let mesh = Mesh::new(cfg);
        assert!(format!("{:?}", mesh).contains("Mesh"));
        assert!(format!("{}", mesh).contains("submesh"));
    }
}
