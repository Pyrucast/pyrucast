//! Mesh — collection of homogeneous submeshes (one element type per
//! submesh).
//!
//! Hierarchy:
//!
//! - [`SubMesh`] — every cell of a single [`ElementType`]. Stores the
//!   connectivity flat (`Vec<NodeId>`, length `cell_count * nodes_per_cell`).
//!   RAII referencing: `add_cell` increments the node refcounts in the
//!   `Coords`; the `SubMesh`'s `Drop` decrements every referenced
//!   node.
//! - [`Mesh`] — aggregate of SubMeshes attached to the same `Coords`.
//!
//! The POI1 case is deliberately degenerate: a POI1 submesh is exactly a
//! list of nodes.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::Coords;
//! use pyrucast::containers::mesh::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::{insert, read};
//!
//! let coords = insert(Coords::new(2).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
//! let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
//!
//! let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
//! sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//! assert_eq!(sm.cell_count(), 1);
//!
//! // The SubMesh holds refs on the 3 nodes, in addition to the `Node`s.
//! assert_eq!(read(&coords).unwrap().refcount(a.id()), 2);
//! drop(sm);  // decrements the referenced nodes
//! assert_eq!(read(&coords).unwrap().refcount(a.id()), 1);
//! ```

pub mod cell;
pub mod color;
pub mod coords;
pub mod element_type;
pub mod node;
pub mod point;

// Flat re-exports: the public types of this module are reachable as
// `mesh::Cell`, `mesh::Coords`, … alongside the `SubMesh` / `Mesh`
// defined here, instead of through their defining sub-module.
pub use cell::{Cell, CellIter};
pub use color::RgbColor;
pub use coords::{Coords, NodeId};
pub use element_type::ElementType;
pub use node::Node;
pub use point::{Point2, Point3, Vector2, Vector3};

use crate::aggregate::Aggregate;
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read, write, Handle};
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
    coords: Handle<Coords>,
    /// Flat connectivity: cell `i` occupies `[i*npc, (i+1)*npc)`.
    connectivity: Vec<NodeId>,
    /// Face colour used by the viz layer. `serde(default)` keeps older
    /// snapshots (without the field) readable.
    #[serde(default)]
    face_color: RgbColor,
}

impl SubMesh {
    /// Create an empty submesh for the given element type, attached to `coords`.
    pub fn new(coords: Handle<Coords>, element_type: ElementType) -> Self {
        Self {
            element_type,
            coords,
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
    /// `Coords`; each node is increfed. On increment failure
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
        {
            let mut c = write(&self.coords)?;
            for (acquired, &n) in nodes.iter().enumerate() {
                if let Err(e) = c.incref(n) {
                    // Roll back the increfs already done for this cell.
                    for &m in &nodes[..acquired] {
                        let _ = c.decref(m);
                    }
                    return Err(e);
                }
            }
        }
        let idx = self.connectivity.len() / npc;
        self.connectivity.extend_from_slice(nodes);
        Ok(idx)
    }

    /// Add a cell whose nodes are **already owned** by the caller (one
    /// refcount unit per node). The SubMesh adopts those units without
    /// increfing further; its `Drop` will decref as usual, which
    /// balances the donation.
    ///
    /// Typical use: a freshly created node (`Coords::add_node`
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
        {
            let c = read(&self.coords)?;
            for &n in nodes {
                if !c.is_alive(n) {
                    return Err(PyrucastError::Message(format!(
                        "add_cell_taking: node {} is not alive",
                        n
                    )));
                }
            }
        }
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

    /// Handle to the owning `Coords` (internal clone).
    pub fn coords(&self) -> Handle<Coords> {
        self.coords.clone()
    }

    /// Build a POI1 submesh with **one cell per [`Node`]**, in the given
    /// order. The [`Coords`] is taken from the nodes themselves
    /// (every [`Node`] carries its own — project convention). Errors if
    /// `nodes` is empty (no Coords to attach to).
    ///
    /// Lower-level form when you already hold the ids and the coords:
    /// [`SubMesh::poi1_from_node_ids`].
    pub fn poi1_from_nodes(nodes: &[Node]) -> Result<SubMesh> {
        let coords = nodes
            .first()
            .ok_or_else(|| {
                PyrucastError::Message("SubMesh::poi1_from_nodes: nodes must not be empty".into())
            })?
            .coords();
        let ids: Vec<NodeId> = nodes.iter().map(|n| n.id()).collect();
        SubMesh::poi1_from_node_ids(coords, &ids)
    }

    /// Build a POI1 submesh with **one cell per node id** in `nodes`, in the
    /// given order. Each node is increfed; on failure the partial submesh's
    /// `Drop` rolls back the increfs already done. The caller is responsible
    /// for any de-duplication (see [`SubMesh::to_poi1`] for the deduped
    /// variant) and supplies the owning `coords` explicitly. When you have
    /// [`Node`] objects, prefer [`SubMesh::poi1_from_nodes`].
    pub fn poi1_from_node_ids(coords: Handle<Coords>, nodes: &[NodeId]) -> Result<SubMesh> {
        let mut sm = SubMesh::new(coords, ElementType::POI1);
        for &nid in nodes {
            sm.add_cell(&[nid])?;
        }
        Ok(sm)
    }

    /// Build a fresh POI1 submesh holding this submesh's nodes,
    /// **de-duplicated in order of first appearance** (one POI1 cell per
    /// unique node). Each referenced node is increfed afresh by the new
    /// submesh; `self` is left untouched.
    ///
    /// Shared building block: [`crate::ops::mesher::to_poi1()`] applies it
    /// submesh-by-submesh, and the physics that need a stable node support
    /// (e.g. heat conduction) use it directly.
    pub fn to_poi1(&self) -> Result<SubMesh> {
        let mut seen: Vec<NodeId> = Vec::new();
        for &nid in &self.connectivity {
            if !seen.contains(&nid) {
                seen.push(nid);
            }
        }
        SubMesh::poi1_from_node_ids(self.coords.clone(), &seen)
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
    /// outer skin (boundary faces) under the painter's algorithm.
    #[cfg(feature = "viz")]
    pub fn plot(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
    ) -> Result<()> {
        self.plot_styled(view, save, crate::viz::MeshStyle::default())
    }

    /// Like [`SubMesh::plot`] but choosing the [`crate::viz::MeshStyle`]:
    /// `Surface` (opaque skin) or `Wireframe` (all edges, see-through).
    #[cfg(feature = "viz")]
    pub fn plot_styled(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        style: crate::viz::MeshStyle,
    ) -> Result<()> {
        crate::viz::render_submesh_styled(self, view, save, style)
    }
}

impl Drop for SubMesh {
    fn drop(&mut self) {
        // One lock acquisition for all decrefs.
        if let Ok(mut c) = write(&self.coords) {
            for &n in &self.connectivity {
                let _ = c.decref(n);
            }
        }
    }
}

impl fmt::Debug for SubMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Bounded structure only — the per-cell connectivity lives in `dump()`.
        f.debug_struct("SubMesh")
            .field("element_type", &self.element_type)
            .field("coords", &self.coords)
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
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
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
/// `Handle<Coords>`; the mesh itself imposes no constraint on
/// `Coords` homogeneity.
#[derive(Serialize, Deserialize, Default)]
pub struct Mesh {
    subs: Vec<Handle<SubMesh>>,
}

crate::impl_aggregate!(Mesh, SubMesh, submesh, "submesh(es)", {
    fn display_extra(&self) -> Option<String> {
        Some(format!(
            ", {} cell(s) total",
            self.cell_count().unwrap_or(0)
        ))
    }
    fn check_push(&self, h: &Handle<SubMesh>) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        let a = self.coords()?;
        let b = read(h)?.coords();
        if a.index() != b.index() || a.generation() != b.generation() {
            Err(PyrucastError::Message("mismatched Coords".into()))
        } else {
            Ok(())
        }
    }
});

crate::impl_aggregate_dump!(Mesh);

// ─── Building POI1 point meshes by union ─────────────────────────────────────
//
// `node.union(node)` and (unitary POI1) `mesh.union_node(node)` both yield a
// fresh unitary POI1 `Mesh` — a points mesh grown one node at a time. Exposed
// to Python as `node | node` and `mesh | node` (the same `|` as the
// aggregates' union). See also [`SubMesh::poi1_from_nodes`].

impl Node {
    /// `node.union(other)` → a unitary POI1 [`Mesh`] over both nodes.
    /// Python: `node | node`.
    pub fn union(&self, other: &Node) -> Result<Mesh> {
        let sm = SubMesh::poi1_from_nodes(&[self.clone(), other.clone()])?;
        Ok(Mesh::from_submesh(sm))
    }
}

impl Mesh {
    /// `mesh.union_node(node)` → a unitary POI1 [`Mesh`] holding this mesh's
    /// points plus `node`. Errors unless `self` is **unitary and POI1**
    /// (exactly one POI1 submesh). Python: `mesh | node`.
    pub fn union_node(&self, node: &Node) -> Result<Mesh> {
        let sub = self.unit()?;
        let (et, coords, mut ids) = {
            let s = read(&sub)?;
            (s.element_type(), s.coords(), s.connectivity().to_vec())
        };
        if et != ElementType::POI1 {
            return Err(PyrucastError::Message(
                "Mesh | Node: expected a unitary POI1 mesh".into(),
            ));
        }
        ids.push(node.id());
        Ok(Mesh::from_submesh(SubMesh::poi1_from_node_ids(
            coords, &ids,
        )?))
    }
}

impl Mesh {
    /// Total cells in the mesh (sum across submeshes).
    pub fn cell_count(&self) -> Result<usize> {
        let mut total = 0usize;
        for sm in self {
            total += read(sm)?.cell_count();
        }
        Ok(total)
    }

    /// Handle to the `Coords` of the first submesh.
    ///
    /// Returns an error if the mesh has no submeshes.
    pub fn coords(&self) -> Result<Handle<Coords>> {
        let sm = self
            .items()
            .first()
            .ok_or_else(|| PyrucastError::Message("coords: mesh has no submeshes".into()))?;
        Ok(read(sm)?.coords())
    }

    /// Create a mesh wrapping a single `SubMesh`. Config-free at the Mesh
    /// level: the submesh already carries its `Coords` (a Mesh is a
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
        write(&self.subs[0])?.add_cell(nodes)
    }

    /// Element type of each submesh, in order.
    pub fn element_types(&self) -> Result<Vec<ElementType>> {
        self.iter().map(|sm| Ok(read(sm)?.element_type())).collect()
    }

    /// Cell count of each submesh, in order.
    pub fn cell_counts(&self) -> Result<Vec<usize>> {
        self.iter().map(|sm| Ok(read(sm)?.cell_count())).collect()
    }

    /// Node at position `node_idx` in cell `cell_idx` of submesh `submesh_idx`.
    pub fn node(&self, submesh_idx: usize, cell_idx: usize, node_idx: usize) -> Result<Node> {
        let sm = self.get(submesh_idx)?;
        let (nid, coords) = {
            let s = read(&sm)?;
            let npc = s.element_type.nodes_per_cell();
            let n = s.cell_count();
            if cell_idx >= n {
                return Err(PyrucastError::Message(format!(
                    "node: cell index {} ≥ cell_count {}",
                    cell_idx, n
                )));
            }
            let nid = s
                .connectivity()
                .get(cell_idx * npc + node_idx)
                .copied()
                .ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "node: node index {} ≥ nodes_per_cell {}",
                        node_idx, npc
                    ))
                })?;
            (nid, s.coords())
        };
        Node::acquire(coords, nid)
    }

    /// Return a `Cell` view on cell `cell_idx` of submesh `submesh_idx`.
    pub fn cell(&self, submesh_idx: usize, cell_idx: usize) -> Result<Cell> {
        let sm = self.get(submesh_idx)?;
        Cell::new(sm, cell_idx)
    }

    /// Iterator over every cell of submesh `submesh_idx`.
    pub fn cells(&self, submesh_idx: usize) -> Result<CellIter> {
        let sm = self.get(submesh_idx)?;
        let end = read(&sm)?.cell_count();
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
        self.plot_styled(view, save, crate::viz::MeshStyle::default())
    }

    /// Like [`Mesh::plot`] but choosing the [`crate::viz::MeshStyle`]:
    /// `Surface` (opaque skin) or `Wireframe` (all edges, see-through).
    /// Each submesh is drawn in its own `face_color`.
    #[cfg(feature = "viz")]
    pub fn plot_styled(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        style: crate::viz::MeshStyle,
    ) -> Result<()> {
        crate::viz::render_mesh_styled(self, view, save, style)
    }

    /// Visualize this mesh coloured by a field component — a
    /// [`crate::containers::node_field::NodeField`] **or** an
    /// [`crate::containers::element_field::ElementField`], uniformly via
    /// [`crate::viz::FieldArg`].
    ///
    /// Per-cell colour comes from the cell's nodal values (read directly
    /// for a node field; fitted per element from the Gauss values for an
    /// element field — inter-element discontinuities stay visible).
    /// `component = None` selects the field's first component.
    ///
    /// The interactive window draws a clickable button at the top
    /// showing the current component and value range; clicking it (or
    /// pressing `Tab`) cycles through the field's components. A colorbar
    /// is drawn on the right edge; `scale` pins its bounds (default:
    /// the data's own min/max).
    ///
    /// For a single submesh, use
    /// [`crate::viz::render_submesh_with_field`] with the submesh handle.
    #[cfg(feature = "viz")]
    pub fn plot_with_field(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        field: crate::viz::FieldArg<'_>,
        component: Option<&str>,
        scale: crate::viz::ColorScale,
        smooth: usize,
    ) -> Result<()> {
        crate::viz::render_mesh_with_field(self, field, component, scale, smooth, view, save)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::insert;

    #[test]
    fn submesh_poi1_is_node_list() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        sm.add_cell(&[b.id()]).unwrap();
        assert_eq!(sm.cell_count(), 2);
        assert_eq!(sm.connectivity()[0], a.id());
        assert_eq!(sm.connectivity()[1], b.id());
    }

    #[test]
    fn poi1_from_nodes_derives_config_and_builds() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        // Node-based form: Coords is taken from the nodes themselves.
        let sm = SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(sm.element_type(), ElementType::POI1);
        assert_eq!(sm.cell_count(), 2);
        assert_eq!(sm.connectivity(), &[a.id(), b.id()]);
        // Matches the id-based form on the same nodes.
        let sm2 = SubMesh::poi1_from_node_ids(coords.clone(), &[a.id(), b.id()]).unwrap();
        assert_eq!(sm.connectivity(), sm2.connectivity());
    }

    #[test]
    fn poi1_from_nodes_empty_is_error() {
        assert!(SubMesh::poi1_from_nodes(&[]).is_err());
    }

    #[test]
    fn submesh_tri3_increfs_and_drop_decrefs() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        // SubMesh increfed each of the 3 nodes, in addition to the Nodes.
        {
            let cf = read(&coords).unwrap();
            assert_eq!(cf.refcount(a.id()), 2);
            assert_eq!(cf.refcount(b.id()), 2);
            assert_eq!(cf.refcount(c.id()), 2);
        }
        drop(sm);
        {
            let cf = read(&coords).unwrap();
            assert_eq!(cf.refcount(a.id()), 1);
            assert_eq!(cf.refcount(b.id()), 1);
            assert_eq!(cf.refcount(c.id()), 1);
        }
    }

    #[test]
    fn submesh_add_cell_invalid_arity() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        let err = sm.add_cell(&[a.id()]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
        // No increment should have survived the failure.
        assert_eq!(read(&coords).unwrap().refcount(a.id()), 1);
    }

    #[test]
    fn submesh_add_cell_collected_node_rollback() {
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let dead_id = write(&coords).unwrap().add_node(&[2.0]).unwrap();
        // dead_id starts at refcount=1; decrement then collect.
        {
            let mut c = write(&coords).unwrap();
            c.decref(dead_id).unwrap();
            assert_eq!(c.gc(), 1);
        }

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        // a (live), b (live), dead_id (collected) → add_cell fails after
        // increfing a and b. The rollback must undo those increfs.
        let err = sm.add_cell(&[a.id(), b.id(), dead_id]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
        {
            let cf = read(&coords).unwrap();
            assert_eq!(cf.refcount(a.id()), 1, "a must be rolled back");
            assert_eq!(cf.refcount(b.id()), 1, "b must be rolled back");
        }
        assert_eq!(sm.cell_count(), 0);
    }

    #[test]
    fn mesh_aggregates_submeshes_same_config() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let cc = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let sm_pts = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm.add_cell(&[b.id()]).unwrap();
            insert(sm)
        };
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), cc.id()]).unwrap();
            insert(sm)
        };

        let mut mesh = Mesh::empty();
        mesh.add_sub(sm_pts).unwrap();
        mesh.add_sub(sm_tri).unwrap();
        assert_eq!(mesh.len(), 2);
        assert_eq!(mesh.cell_count().unwrap(), 3); // 2 points + 1 triangle
    }

    #[test]
    fn mesh_element_types_and_cell_counts() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        m.add_cell(&[b.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
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
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            insert(sm)
        };
        m.add_sub(sm_tri).unwrap();

        let et0 = read(&m[0]).unwrap().element_type();
        let et1 = read(&m[1]).unwrap().element_type();
        assert_eq!(et0, ElementType::POI1);
        assert_eq!(et1, ElementType::TRI3);

        let types: Vec<ElementType> = (&m)
            .into_iter()
            .map(|h| read(h).unwrap().element_type())
            .collect();
        assert_eq!(types, vec![ElementType::POI1, ElementType::TRI3]);
    }

    #[test]
    fn mesh_node_access_by_indices() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let n = m.node(0, 0, 0).unwrap();
        assert_eq!(n.id(), a.id()); // node 0 of element 0 = a
        assert!(m.node(1, 0, 0).is_err()); // submesh out of bounds
        assert!(m.node(0, 1, 0).is_err()); // cell out of bounds
        assert!(m.node(0, 0, 3).is_err()); // node out of bounds (TRI3: indices 0..2)
    }

    #[test]
    fn mesh_merge_combines_submeshes() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut m1 = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m1.add_cell(&[a.id()]).unwrap();
        m1.add_cell(&[b.id()]).unwrap();

        let mut m2 = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        m2.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

        let merged = m1.union(&m2).unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.cell_count().unwrap(), 3); // 2 POI1 + 1 TRI3
    }

    #[test]
    fn debug_and_display_submesh_and_mesh() {
        let coords = insert(Coords::new(1).unwrap());
        let sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        let d = format!("{:?}", sm);
        let s = format!("{}", sm);
        assert!(d.contains("SubMesh"));
        assert!(s.contains("SEG2"));

        let mesh = Mesh::empty();
        assert!(format!("{:?}", mesh).contains("Mesh"));
        assert!(format!("{}", mesh).contains("submesh"));
    }

    #[test]
    fn aggregate_union_sub_and_sub_union_sub() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let s1 = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let s2 = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&b)).unwrap());

        // sub | sub → Mesh
        let m = Mesh::union_subs(&s1, &s2).unwrap();
        assert_eq!(m.len(), 2);

        // aggregate | sub → Mesh
        let s3 = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let m2 = m.union_sub(&s3).unwrap();
        assert_eq!(m2.len(), 3);
    }

    #[test]
    fn node_union_node_and_mesh_union_node() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();

        let m = a.union(&b).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(read(&m.unit().unwrap()).unwrap().cell_count(), 2);

        let m2 = m.union_node(&c).unwrap();
        assert_eq!(read(&m2.unit().unwrap()).unwrap().cell_count(), 3);
    }

    #[test]
    fn mesh_union_node_rejects_non_unitary_poi1() {
        let coords = insert(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        // Non-POI1 → error.
        assert!(tri.union_node(&a).is_err());
    }
}
