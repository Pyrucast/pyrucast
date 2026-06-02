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
//! use pyrucast::containers::mesh::Configuration;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::containers::mesh::Node;
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

pub mod cell;
pub mod color;
pub mod configuration;
pub mod element_type;
pub mod node;
pub mod point;

// Flat re-exports: the public types of this module are reachable as
// `mesh::Cell`, `mesh::Configuration`, … alongside the `SubMesh` / `Mesh`
// defined here, instead of through their defining sub-module.
pub use cell::{Cell, CellIter};
pub use color::RgbColor;
pub use configuration::{Configuration, NodeId};
pub use element_type::ElementType;
pub use node::Node;
pub use point::{Point2, Point3, Vector2, Vector3};

use crate::aggregate::Aggregate;
use crate::error::{PyrucastError, Result};
use crate::store::{insert, with, with_mut, Handle};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── SubMesh ────────────────────────────────────────────────────────────────

/// Submesh: every cell of a single [`ElementType`].
///
/// The connectivity is stored flat; each cell occupies
/// `element_type.nodes_per_cell()` contiguous entries.
///
/// A [`RgbColor`] is attached as the **face colour** used by the
/// visualization layer (`viz` feature); it has no effect on numerics and
/// defaults to a light blue.
#[derive(Serialize, Deserialize)]
pub struct SubMesh {
    element_type: ElementType,
    config: Handle<Configuration>,
    /// Flat connectivity: cell `i` occupies `[i*npc, (i+1)*npc)`.
    connectivity: Vec<NodeId>,
    /// Face colour used by the viz layer. `serde(default)` keeps older
    /// snapshots (without the field) readable.
    #[serde(default)]
    face_color: RgbColor,
}

impl SubMesh {
    /// Create an empty submesh for the given element type, attached to `config`.
    pub fn new(config: Handle<Configuration>, element_type: ElementType) -> Self {
        Self {
            element_type,
            config,
            connectivity: Vec::new(),
            face_color: RgbColor::default(),
        }
    }

    /// Face colour used when this submesh is drawn (no numerical effect).
    pub fn face_color(&self) -> RgbColor {
        self.face_color
    }

    /// Replace the face colour used when this submesh is drawn.
    pub fn set_face_color(&mut self, color: RgbColor) {
        self.face_color = color;
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

    /// Add a cell whose nodes are **already owned** by the caller (one
    /// refcount unit per node). The SubMesh adopts those units without
    /// increfing further; its `Drop` will decref as usual, which
    /// balances the donation.
    ///
    /// Typical use: a freshly created node (`Configuration::add_node`
    /// returns refcount = 1) is handed directly to a POI1 SubMesh which
    /// then becomes its sole owner.
    ///
    /// The caller is responsible for the ownership claim; this method
    /// only checks that the cell length matches the element type and
    /// that the nodes are alive at the moment of the call.
    pub fn add_cell_taking(&mut self, nodes: &[NodeId]) -> Result<usize> {
        let npc = self.element_type.nodes_per_cell();
        if nodes.len() != npc {
            return Err(PyrucastError::Message(format!(
                "add_cell_taking({}): expected {} nodes, got {}",
                self.element_type,
                npc,
                nodes.len()
            )));
        }
        with(&self.config, |c| -> Result<()> {
            for &n in nodes {
                if !c.is_alive(n) {
                    return Err(PyrucastError::Message(format!(
                        "add_cell_taking: node {} is not alive",
                        n
                    )));
                }
            }
            Ok(())
        })??;
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

    /// Visualize this submesh.
    ///
    /// - `view = None` ⇒ [`crate::viz::View::default`] (isometric).
    /// - `save = None` ⇒ open an interactive window (requires feature
    ///   `viz-interactive`).
    /// - `save = Some(path)` ⇒ write an image file; the format is inferred
    ///   from the extension (`.png` or `.svg`).
    ///
    /// Every supported element type is rendered: POI1 as dots, SEG2 as
    /// segments, TRI3 / QUA4 as filled polygons, and TET4 / HEX8 as their
    /// outer faces (4 triangles or 6 quads) under the painter's algorithm.
    #[cfg(feature = "viz")]
    pub fn plot(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
    ) -> Result<()> {
        crate::viz::render(self, view, save)
    }

    /// Visualize this submesh coloured by a [`crate::containers::node_field::NodeField`]
    /// component.
    ///
    /// Per-cell colour is the mean of the field's component at the cell's
    /// nodes, mapped through a blue → green → red colormap. Nodes
    /// absent from the field's support are ignored in the mean.
    /// `component = None` selects the field's first component.
    ///
    /// The interactive window draws a clickable button at the top
    /// showing the current component and value range; clicking it (or
    /// pressing `Tab`) cycles through the field's components. A colorbar
    /// is drawn on the right edge; `scale` pins its bounds (default:
    /// the data's own min/max).
    #[cfg(feature = "viz")]
    pub fn plot_with_field(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        field: &crate::containers::node_field::NodeField,
        component: Option<&str>,
        scale: crate::viz::ColorScale,
    ) -> Result<()> {
        crate::viz::render_submesh_with_field(self, field, component, scale, view, save)
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
        // Bounded structure only — the per-cell connectivity lives in `dump()`.
        f.debug_struct("SubMesh")
            .field("element_type", &self.element_type)
            .field("configuration", &self.config)
            .field("cell_count", &self.cell_count())
            .field("face_color", &self.face_color)
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

impl crate::dump::Dump for SubMesh {
    fn dump_with(&self, opts: &crate::dump::DumpOptions) -> String {
        use crate::dump::table;
        let npc = self.element_type.nodes_per_cell();
        let mut headers = vec!["cell".to_string()];
        headers.extend((0..npc).map(|i| format!("n{i}")));
        let rows: Vec<Vec<String>> = if npc > 0 {
            self.connectivity
                .chunks(npc)
                .enumerate()
                .map(|(i, chunk)| {
                    let mut row = vec![i.to_string()];
                    row.extend(chunk.iter().map(|nid| nid.to_string()));
                    row
                })
                .collect()
        } else {
            Vec::new()
        };
        format!("{self}\n{}", table(&headers, &rows, opts))
    }
}

// ─── Mesh ───────────────────────────────────────────────────────────────────

/// Mesh: aggregate of submeshes. Each submesh carries its own
/// `Handle<Configuration>`; the mesh itself imposes no constraint on
/// configuration homogeneity.
#[derive(Serialize, Deserialize, Default)]
pub struct Mesh {
    subs: Vec<Handle<SubMesh>>,
}

crate::impl_aggregate!(Mesh, SubMesh, submesh, "submesh(es)", {
    fn display_extra(&self) -> Option<String> {
        Some(format!(", {} cell(s) total", self.cell_count().unwrap_or(0)))
    }
    fn check_push(&self, h: &Handle<SubMesh>) -> Result<()> {
        if self.is_empty() { return Ok(()); }
        let a = self.configuration()?;
        let b = with(h, |s| s.configuration())?;
        if a.index() != b.index() || a.generation() != b.generation() {
            Err(PyrucastError::Message("mismatched Configurations".into()))
        } else {
            Ok(())
        }
    }
});

impl Mesh {
    /// Total cells in the mesh (sum across submeshes).
    pub fn cell_count(&self) -> Result<usize> {
        let mut total = 0usize;
        for sm in self {
            total += with(sm, |s| s.cell_count())?;
        }
        Ok(total)
    }

    /// Handle to the `Configuration` of the first submesh.
    ///
    /// Returns an error if the mesh has no submeshes.
    pub fn configuration(&self) -> Result<Handle<Configuration>> {
        let sm = self.items().first().ok_or_else(|| {
            PyrucastError::Message("configuration: mesh has no submeshes".into())
        })?;
        with(sm, |s| s.configuration())
    }

    /// Create a mesh wrapping a single `SubMesh`. Config-free at the Mesh
    /// level: the submesh already carries its `Configuration` (a Mesh is a
    /// pure aggregate of submeshes). The submesh is moved into the store.
    pub fn from_submesh(sub: SubMesh) -> Self {
        let mut mesh = Self::default();
        mesh.subs.push(insert(sub));
        mesh
    }

    /// Add a cell directly when the mesh has exactly one submesh.
    pub fn add_cell(&mut self, nodes: &[NodeId]) -> Result<usize> {
        if self.len() != 1 {
            return Err(PyrucastError::Message(
                "add_cell: mesh must have exactly one submesh".into(),
            ));
        }
        with_mut(&self.subs[0], |s| s.add_cell(nodes))?
    }

    /// Element type of each submesh, in order.
    pub fn element_types(&self) -> Result<Vec<ElementType>> {
        self.iter()
            .map(|sm| with(sm, |s| s.element_type()))
            .collect()
    }

    /// Cell count of each submesh, in order.
    pub fn cell_counts(&self) -> Result<Vec<usize>> {
        self.iter()
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
        let cfg = with(&sm, |s| s.configuration())?;
        Node::acquire(cfg, nid)
    }

    /// Return a `Cell` view on cell `cell_idx` of submesh `submesh_idx`.
    pub fn cell(&self, submesh_idx: usize, cell_idx: usize) -> Result<Cell> {
        let sm = self.submesh(submesh_idx)?;
        Cell::new(sm, cell_idx)
    }

    /// Iterator over every cell of submesh `submesh_idx`.
    pub fn cells(&self, submesh_idx: usize) -> Result<CellIter> {
        let sm = self.submesh(submesh_idx)?;
        let end = with(&sm, |s| s.cell_count())?;
        Ok(CellIter::new(sm, end))
    }

    /// Visualize this mesh — every submesh is drawn, each in its own
    /// [`SubMesh::face_color`]. See [`SubMesh::plot`] for the meaning of
    /// `view` and `save` and the supported element types.
    #[cfg(feature = "viz")]
    pub fn plot(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
    ) -> Result<()> {
        crate::viz::render(self, view, save)
    }

    /// Visualize this mesh coloured by a [`crate::containers::node_field::NodeField`]
    /// component. See [`SubMesh::plot_with_field`] for the meaning of
    /// `view`, `save`, `field` and `component`. In the interactive
    /// window the same component button is drawn over the whole mesh
    /// and cycles through every component of `field`.
    #[cfg(feature = "viz")]
    pub fn plot_with_field(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        field: &crate::containers::node_field::NodeField,
        component: Option<&str>,
        scale: crate::viz::ColorScale,
    ) -> Result<()> {
        crate::viz::render_mesh_with_field(self, field, component, scale, view, save)
    }

}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
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

        let mut mesh = Mesh::empty();
        mesh.add_sub(sm_pts).unwrap();
        mesh.add_sub(sm_tri).unwrap();
        assert_eq!(mesh.submesh_count(), 2);
        assert_eq!(mesh.cell_count().unwrap(), 3); // 2 points + 1 triangle
    }

    #[test]
    fn mesh_element_types_and_cell_counts() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        m.add_cell(&[b.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        m.add_sub(sm_tri).unwrap();

        assert_eq!(
            m.element_types().unwrap(),
            vec![ElementType::POI1, ElementType::TRI3]
        );
        assert_eq!(m.cell_counts().unwrap(), vec![2, 1]);
    }

    #[test]
    fn mesh_index_and_iter_sugar() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(cfg.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        m.add_sub(sm_tri).unwrap();

        let et0 = with(&m[0], |s| s.element_type()).unwrap();
        let et1 = with(&m[1], |s| s.element_type()).unwrap();
        assert_eq!(et0, ElementType::POI1);
        assert_eq!(et1, ElementType::TRI3);

        let types: Vec<ElementType> = (&m)
            .into_iter()
            .map(|h| with(h, |s| s.element_type()).unwrap())
            .collect();
        assert_eq!(types, vec![ElementType::POI1, ElementType::TRI3]);
    }

    #[test]
    fn mesh_node_access_by_indices() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let n = m.node(0, 0, 0).unwrap();
        assert_eq!(n.id(), a.id()); // node 0 of element 0 = a
        assert!(m.node(1, 0, 0).is_err()); // submesh out of bounds
        assert!(m.node(0, 1, 0).is_err()); // cell out of bounds
        assert!(m.node(0, 0, 3).is_err()); // node out of bounds (TRI3: indices 0..2)
    }

    #[test]
    fn mesh_merge_combines_submeshes() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.5, 1.0]).unwrap();

        let mut m1 = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::POI1));
        m1.add_cell(&[a.id()]).unwrap();
        m1.add_cell(&[b.id()]).unwrap();

        let mut m2 = Mesh::from_submesh(SubMesh::new(cfg.clone(), ElementType::TRI3));
        m2.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let merged = (&m1 + &m2).unwrap();
        assert_eq!(merged.submesh_count(), 2);
        assert_eq!(merged.cell_count().unwrap(), 3); // 2 POI1 + 1 TRI3
    }

    #[test]
    fn debug_and_display_submesh_and_mesh() {
        let cfg = insert(Configuration::new(1).unwrap());
        let sm = SubMesh::new(cfg.clone(), ElementType::SEG2);
        let d = format!("{:?}", sm);
        let s = format!("{}", sm);
        assert!(d.contains("SubMesh"));
        assert!(s.contains("SEG2"));

        let mesh = Mesh::empty();
        assert!(format!("{:?}", mesh).contains("Mesh"));
        assert!(format!("{}", mesh).contains("submesh"));
    }
}
