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
//! use pyrucast::coords::Coords;
//! use pyrucast::atoms::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::atoms::Node;
//! use pyrucast::handle::Handle;
//!
//! let coords = Handle::new(Coords::new(2).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
//! let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
//!
//! let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
//! sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
//! assert_eq!(sm.cell_count(), 1);
//!
//! // The SubMesh holds refs on the 3 nodes, in addition to the `Node`s.
//! assert_eq!(coords.read().refcount(a.id()), 2);
//! drop(sm);  // decrements the referenced nodes
//! assert_eq!(coords.read().refcount(a.id()), 1);
//! ```

use crate::aggregate::Aggregate;
use crate::atoms::{Cell, CellIter, ElementType, Node, NodeId, RgbColor};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::OnceLock;

// ─── SubMesh ────────────────────────────────────────────────────────────────

/// Submesh: every cell of a single [`ElementType`].
///
/// The connectivity is stored flat; each cell occupies
/// `element_type.nodes_per_cell()` contiguous entries.
///
/// A [`RgbColor`] is attached as the **face colour** used by the
/// visualization layer (`viz` feature); it has no effect on numerics and
/// defaults to a light blue.
pub struct SubMesh {
    element_type: ElementType,
    coords: Handle<Coords>,
    /// Flat connectivity: cell `i` occupies `[i*npc, (i+1)*npc)`.
    connectivity: Vec<NodeId>,
    /// Face colour used by the viz layer. `serde(default)` keeps older
    /// snapshots (without the field) readable.
    face_color: RgbColor,
    /// Once **sealed**, the connectivity is frozen: [`SubMesh::add_cell`] and
    /// [`SubMesh::add_cell_taking`] refuse to run. A submesh is sealed the
    /// first time a non-mesh consumer (finite-element space, field, matrix, …)
    /// captures its handle, so those consumers can never be left referencing
    /// stale cells. The seal is permanent for the object's lifetime.
    /// `serde(default)` keeps older snapshots (without the field) readable.
    sealed: bool,
    /// Lazily-built `NodeId → index` map over the **distinct** nodes of the
    /// connectivity, in first-appearance order. Consumers that need a node
    /// lookup (node fields, …) read it in place while holding their store
    /// guard on this submesh — no copy — so the O(n) build is paid once and
    /// mutualised across every field on this support. Not serialized — it is
    /// derived from `connectivity` and rebuilt on demand after a reload. Only
    /// ever populated once the submesh is sealed (its connectivity frozen),
    /// so it can never go stale.
    node_index: OnceLock<HashMap<NodeId, usize>>,
    /// Lazily-built **canonical POI1 companion**: the node cloud of this
    /// submesh's distinct nodes, materialised once and shared. Every consumer
    /// that projects this submesh to its nodes ([`SubMesh::to_poi1`]) gets the
    /// *same* store slot, so their node fields pair under
    /// [`same_support`](crate::containers::field::SubField::same_support) — a
    /// stiffness block's support, a `restrict` onto this mesh, and a
    /// `divergence`/`flux` output over it all land on one handle and combine
    /// directly. Not serialized (derived from `connectivity`, rebuilt on
    /// demand). Only ever populated once the submesh is sealed, so it can never
    /// go stale.
    poi1_companion: OnceLock<Handle<SubMesh>>,
}

impl SubMesh {
    /// Create an empty submesh for the given element type, attached to `coords`.
    pub fn new(coords: Handle<Coords>, element_type: ElementType) -> Self {
        Self {
            element_type,
            coords,
            connectivity: Vec::new(),
            face_color: RgbColor::default(),
            sealed: false,
            node_index: OnceLock::new(),
            poi1_companion: OnceLock::new(),
        }
    }

    /// Whether this submesh is sealed (connectivity frozen).
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// `NodeId → index` map over the **distinct** nodes of the connectivity,
    /// in first-appearance order (the order [`SubNodeField`](crate::containers::node_field::SubNodeField) snapshots its
    /// support in). Built once and cached; callers keep their read guard on
    /// this submesh while using the returned reference — no copy.
    ///
    /// Meant for sealed supports: the map is derived from `connectivity`, and
    /// a sealed submesh can no longer grow, so the cache can never go stale.
    /// (It is only ever queried through a sealed support in practice.)
    pub fn node_index(&self) -> &HashMap<NodeId, usize> {
        self.node_index.get_or_init(|| {
            let mut map = HashMap::with_capacity(self.connectivity.len());
            for &nid in &self.connectivity {
                let next = map.len();
                map.entry(nid).or_insert(next);
            }
            map
        })
    }

    /// Seal this submesh: freeze its connectivity permanently. After this,
    /// [`SubMesh::add_cell`] / [`SubMesh::add_cell_taking`] return
    /// [`PyrucastError::MeshSealed`]. Idempotent.
    ///
    /// Called by the container layer whenever a non-mesh object captures the
    /// submesh — see the free function [`seal`]. A bare [`Mesh`] holding the
    /// submesh does **not** seal it (a mesh may keep growing until consumed).
    pub fn seal(&mut self) {
        self.sealed = true;
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
        if self.sealed {
            return Err(PyrucastError::MeshSealed);
        }
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
            let mut c = self.coords.write();
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
        if self.sealed {
            return Err(PyrucastError::MeshSealed);
        }
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
            let c = self.coords.read();
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

    /// Deep-copy this submesh into a **fresh, unsealed** one: same element
    /// type, same `Coords`, same connectivity (each referenced node increfed
    /// anew) and same face colour — but never inheriting the seal.
    ///
    /// This is the escape hatch for the seal: once a consumer has frozen a
    /// submesh, `duplicate()` hands back an independent copy you can keep
    /// editing with [`SubMesh::add_cell`]. The two share the same `Coords`
    /// (nodes are not cloned, only their refcounts bumped).
    pub fn duplicate(&self) -> Result<SubMesh> {
        let mut copy = SubMesh::new(self.coords.clone(), self.element_type);
        copy.face_color = self.face_color;
        let npc = self.element_type.nodes_per_cell();
        if npc > 0 {
            for chunk in self.connectivity.chunks(npc) {
                // `copy` is unsealed, so `add_cell` runs and increfs the
                // nodes (with rollback on failure).
                copy.add_cell(chunk)?;
            }
        }
        Ok(copy)
    }

    /// Rename this submesh's nodes **in place**, through `map`.
    ///
    /// Every node id that `map` mentions is replaced by its image wherever it
    /// appears in the connectivity; ids absent from the map (and images equal
    /// to their key) are left alone. Returns the number of rewritten slots.
    ///
    /// This is a **renaming**, not an edit of the mesh structure: the element
    /// type, the number of cells and the cell order are untouched, so every
    /// index a caller holds on this submesh (cell numbers, and therefore the
    /// element fields keyed on them) stays valid. That is what makes an
    /// in-place rewrite defensible on a container that otherwise only ever
    /// grows — it is the seam [`merge_nodes(…, in_place = true)`](fn@crate::ops::mesh::merge_nodes)
    /// welds shared meshes through.
    ///
    /// Refcounts follow the rename: each rewritten slot increfs its new node
    /// and decrefs the old one (the connectivity owns one unit per
    /// *occurrence*). Nothing is written unless every incref succeeds, so a
    /// dead or invalid image leaves the submesh exactly as it was. The lazily
    /// built caches ([`node_index`](SubMesh::node_index),
    /// [`to_poi1`](SubMesh::to_poi1)'s companion) are derived from the
    /// connectivity and are therefore dropped.
    ///
    /// Refuses with [`PyrucastError::MeshSealed`] on a **sealed** submesh: a
    /// consumer (finite-element space, field, matrix) has captured it, and its
    /// node numbering must not move under it. Use
    /// [`duplicate`](SubMesh::duplicate) to get an editable copy.
    ///
    /// Node **positions** are never touched — this only rewrites which node a
    /// cell refers to.
    pub fn remap_nodes(&mut self, map: &HashMap<NodeId, NodeId>) -> Result<usize> {
        if self.sealed {
            return Err(PyrucastError::MeshSealed);
        }

        // Slots to rewrite, with the refcount move each one implies.
        let changes: Vec<(usize, NodeId, NodeId)> = self
            .connectivity
            .iter()
            .enumerate()
            .filter_map(|(i, &old)| match map.get(&old) {
                Some(&new) if new != old => Some((i, old, new)),
                _ => None,
            })
            .collect();
        if changes.is_empty() {
            return Ok(0);
        }

        {
            let mut c = self.coords.write();
            // Incref the images first: until they have all succeeded, nothing
            // has been given up, so a rollback restores the initial state.
            for (done, &(_, _, new)) in changes.iter().enumerate() {
                if let Err(e) = c.incref(new) {
                    for &(_, _, rollback) in &changes[..done] {
                        let _ = c.decref(rollback);
                    }
                    return Err(e);
                }
            }
            for &(_, old, _) in &changes {
                c.decref(old)?;
            }
        }

        for &(i, _, new) in &changes {
            self.connectivity[i] = new;
        }
        // Both caches are derived from the connectivity that just moved.
        self.node_index = OnceLock::new();
        self.poi1_companion = OnceLock::new();
        Ok(changes.len())
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

    /// Canonical POI1 node cloud of this submesh — its nodes **de-duplicated in
    /// order of first appearance** (one POI1 cell per unique node), as a sealed
    /// [`SubMesh`] handle.
    ///
    /// **Cached, per submesh.** Once `self` is sealed the companion is built at
    /// most once and every later call returns the *same* store slot, so all the
    /// node fields that project this submesh to its nodes pair under
    /// [`same_support`](crate::containers::field::SubField::same_support): a
    /// stiffness block's support (built this way in every physics' `new`), a
    /// [`restrict`](fn@crate::ops::node_field::restrict) onto this mesh, and a
    /// `divergence`/`flux`/`internal_forces` output over it all share one handle
    /// and combine directly by the field operators. This is what lets
    /// `solve(K, f) - restrict(g, mesh)` and `&K * &restrict(f, mesh)` line up.
    ///
    /// On an **unsealed** submesh nothing is cached — a fresh cloud is returned
    /// each call (the old behaviour), since the connectivity could still change
    /// and stale the companion. Shared building block:
    /// [`crate::ops::mesh::to_poi1()`] applies it submesh-by-submesh.
    pub fn to_poi1(&self) -> Result<Handle<SubMesh>> {
        if let Some(h) = self.poi1_companion.get() {
            return Ok(h.clone());
        }
        // De-duplicate in **order of first appearance**. The membership test goes
        // through a hash set, not a linear scan of `seen`: this runs on every
        // model construction, and a `Vec::contains` here made the whole operator
        // quadratic (640 k QUA4 took ~8 min, versus ~40 ms now).
        let mut seen: Vec<NodeId> = Vec::with_capacity(self.connectivity.len());
        let mut known: HashSet<NodeId> = HashSet::with_capacity(self.connectivity.len());
        for &nid in &self.connectivity {
            if known.insert(nid) {
                seen.push(nid);
            }
        }
        // Build (write-locks `Coords`) and seal the companion. `self` is behind
        // the caller's read guard on this submesh — a different slot than the
        // POI1 companion and `Coords` — so no lock inversion (same discipline
        // the previous `Handle::new(sm.read().to_poi1()?)` idiom already relied on).
        let handle = Handle::new(SubMesh::poi1_from_node_ids(self.coords.clone(), &seen)?);
        seal(&handle)?;
        if self.sealed {
            // Frozen source ⇒ safe to memoize. On a race the loser drops its
            // build and everyone reads the winner's slot.
            let _ = self.poi1_companion.set(handle);
            Ok(self
                .poi1_companion
                .get()
                .expect("populated on this path")
                .clone())
        } else {
            Ok(handle)
        }
    }

    /// Visualize this submesh.
    ///
    /// - `view = None` ⇒ [`crate::viz::View::default`] (isometric).
    /// - `save = None` ⇒ open an interactive window (requires feature
    ///   `viz-interactive`).
    /// - `save = Some(path)` ⇒ write an image file; the format is inferred
    ///   from the extension (`.png`, `.svg`, or `.svgz` for the same SVG
    ///   gzipped — around a tenth of the bytes on disk).
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
        self.plot_styled(view, save, crate::viz::MeshStyle::default(), None)
    }

    /// Like [`SubMesh::plot`] but choosing the [`crate::viz::MeshStyle`]:
    /// `Surface` (opaque skin) or `Wireframe` (all edges, see-through).
    /// `title`, if given, names the interactive window and is drawn as a
    /// caption at the bottom of a saved PNG/SVG.
    #[cfg(feature = "viz")]
    pub fn plot_styled(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        style: crate::viz::MeshStyle,
        title: Option<&str>,
    ) -> Result<()> {
        crate::viz::render_submesh_styled(self, view, save, style, title)
    }
}

/// Seal the submesh behind `handle`, freezing its connectivity.
///
/// This is the seam every non-mesh consumer goes through when it captures a
/// [`SubMesh`] handle (finite-element space, node field, matrix support, …):
/// from that point on the submesh can no longer grow, so the consumer's
/// cell-indexed view can never go stale. Idempotent; returns the same handle
/// (cloned) for ergonomic chaining at a constructor's capture site.
///
/// **Already-sealed fast path takes only a read lock.** This matters now that
/// [`SubMesh::to_poi1`] is cached: a `restrict` onto a mesh can land on the very
/// support a source field already sits on (the shared POI1 companion), so
/// `from_poi1` may `seal` a support while the caller still holds a *read* guard
/// on it (a field `view`). Sealing is idempotent, so when the submesh is already
/// sealed we skip the write entirely — a read lock coexists with that reader,
/// whereas a write lock would deadlock against it. Taking a write lock while a
/// **write** guard on the same slot is held is still a deadlock (the slot lock is
/// not reentrant — see [`crate::handle`]); only the sealed-read case is relaxed.
pub fn seal(handle: &Handle<SubMesh>) -> Result<Handle<SubMesh>> {
    if handle.read().is_sealed() {
        return Ok(handle.clone());
    }
    handle.write().seal();
    Ok(handle.clone())
}

impl Drop for SubMesh {
    fn drop(&mut self) {
        // One lock acquisition for all decrefs.
        let mut c = self.coords.write();
        for &n in &self.connectivity {
            let _ = c.decref(n);
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
            .field("sealed", &self.sealed)
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
#[derive(Default)]
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
        let b = h.read().coords();
        if !a.same_object(&b) {
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
            let s = sub.read();
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
            total += sm.read().cell_count();
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
        Ok(sm.read().coords())
    }

    /// Create a mesh wrapping a single `SubMesh`. Config-free at the Mesh
    /// level: the submesh already carries its `Coords` (a Mesh is a
    /// pure aggregate of submeshes). The submesh is moved into the store.
    pub fn from_submesh(sub: SubMesh) -> Self {
        let mut mesh = Self::default();
        mesh.subs.push(Handle::new(sub));
        mesh
    }

    /// Deep-copy the whole mesh: every submesh is [`SubMesh::duplicate`]d
    /// into a fresh, unsealed submesh under a new handle. The copy is fully
    /// editable even when the source's submeshes have been sealed by their
    /// consumers; nodes are shared (same `Coords`), only their refcounts grow.
    pub fn duplicate(&self) -> Result<Mesh> {
        let mut copy = Self::default();
        for sm in self {
            let dup = sm.read().duplicate()?;
            copy.subs.push(Handle::new(dup));
        }
        Ok(copy)
    }

    /// Add a cell directly when the mesh has exactly one submesh.
    pub fn add_cell(&mut self, nodes: &[NodeId]) -> Result<usize> {
        if self.len() != 1 {
            return Err(PyrucastError::Message(
                "add_cell: mesh must have exactly one submesh".into(),
            ));
        }
        self.subs[0].write().add_cell(nodes)
    }

    /// Element type of each submesh, in order.
    pub fn element_types(&self) -> Result<Vec<ElementType>> {
        self.iter().map(|sm| Ok(sm.read().element_type())).collect()
    }

    /// Cell count of each submesh, in order.
    pub fn cell_counts(&self) -> Result<Vec<usize>> {
        self.iter().map(|sm| Ok(sm.read().cell_count())).collect()
    }

    /// A POI1 mesh holding exactly `nodes` — the named constructor of a point
    /// cloud from atoms.
    ///
    /// The parent-level form of
    /// [`SubMesh::poi1_from_nodes`](SubMesh::poi1_from_nodes), which the
    /// aggregate rule asks for: a named constructor lives on the parent and
    /// returns a parent.
    pub fn poi1_from_nodes(nodes: &[Node]) -> Result<Mesh> {
        Ok(Mesh::from_submesh(SubMesh::poi1_from_nodes(nodes)?))
    }

    /// The mesh node closest (Euclidean distance) to `point`.
    ///
    /// `point` must have the mesh `Coords` spatial dimension. Only nodes
    /// actually referenced by a cell are considered; ties are broken by the
    /// smaller `NodeId`, so the result does not depend on iteration order.
    ///
    /// The natural way to pick a node to pin a boundary condition on, or to
    /// read a result at, when you know roughly *where* it is but not its id.
    ///
    /// Errors if the mesh has no submeshes, references no nodes, or if
    /// `point`'s length does not match the coordinate dimension.
    pub fn nearest_node(&self, point: &[f64]) -> Result<Node> {
        let coords_handle = self.coords()?;

        // Gather the unique node ids the mesh references, across all submeshes.
        let mut seen: HashSet<NodeId> = HashSet::new();
        for sm in self {
            let s = sm.read();
            for &nid in s.connectivity() {
                seen.insert(nid);
            }
        }

        let best = {
            let c = coords_handle.read();
            if point.len() != c.dim() as usize {
                return Err(PyrucastError::Message(format!(
                    "nearest_node: point has {} coordinates, mesh is {}-D",
                    point.len(),
                    c.dim()
                )));
            }
            let mut best: Option<(NodeId, f64)> = None;
            for &nid in &seen {
                let x = c.position(nid)?;
                let d2: f64 = x.iter().zip(point).map(|(a, b)| (a - b) * (a - b)).sum();
                // Strictly-less keeps the first (smallest id) on a tie, but `seen`
                // is a set with no stable order, so compare ids explicitly.
                match best {
                    Some((bid, bd2)) if bd2 < d2 || (bd2 == d2 && bid.0 <= nid.0) => {}
                    _ => best = Some((nid, d2)),
                }
            }
            best
        };

        let (nid, _) = best.ok_or_else(|| {
            PyrucastError::Message("nearest_node: mesh references no nodes".into())
        })?;
        Node::acquire(coords_handle, nid)
    }

    /// Node at position `node_idx` in cell `cell_idx` of submesh `submesh_idx`.
    pub fn node(&self, submesh_idx: usize, cell_idx: usize, node_idx: usize) -> Result<Node> {
        let sm = self.get(submesh_idx)?;
        let (nid, coords) = {
            let s = sm.read();
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
        let end = sm.read().cell_count();
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
        self.plot_styled(view, save, crate::viz::MeshStyle::default(), None)
    }

    /// Like [`Mesh::plot`] but choosing the [`crate::viz::MeshStyle`]:
    /// `Surface` (opaque skin) or `Wireframe` (all edges, see-through).
    /// Each submesh is drawn in its own `face_color`. `title`, if given,
    /// names the interactive window and is drawn as a caption at the bottom
    /// of a saved PNG/SVG.
    #[cfg(feature = "viz")]
    pub fn plot_styled(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        style: crate::viz::MeshStyle,
        title: Option<&str>,
    ) -> Result<()> {
        crate::viz::render_mesh_styled(self, view, save, style, title)
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
    // Eight orthogonal rendering options, all optional at the Python layer:
    // grouping them into a struct would only move the argument list.
    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "viz")]
    pub fn plot_with_field(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        field: crate::viz::FieldArg<'_>,
        component: Option<&str>,
        scale: crate::viz::ColorScale,
        smooth: usize,
        title: Option<&str>,
    ) -> Result<()> {
        crate::viz::render_mesh_with_field(self, field, component, scale, smooth, view, save, title)
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;

    #[test]
    fn submesh_poi1_is_node_list() {
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        // SubMesh increfed each of the 3 nodes, in addition to the Nodes.
        {
            let cf = coords.read();
            assert_eq!(cf.refcount(a.id()), 2);
            assert_eq!(cf.refcount(b.id()), 2);
            assert_eq!(cf.refcount(c.id()), 2);
        }
        drop(sm);
        {
            let cf = coords.read();
            assert_eq!(cf.refcount(a.id()), 1);
            assert_eq!(cf.refcount(b.id()), 1);
            assert_eq!(cf.refcount(c.id()), 1);
        }
    }

    #[test]
    fn node_index_maps_distinct_nodes_in_first_appearance_order() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();

        // Two QUA4 cells sharing the edge (b, c): b and c appear twice.
        let mut sm = SubMesh::new(coords.clone(), ElementType::QUA4);
        sm.add_cell(&[a.id(), b.id(), c.id(), d.id()]).unwrap();
        let e = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
        let f = Node::create_in(coords.clone(), &[2.0, 1.0]).unwrap();
        sm.add_cell(&[b.id(), e.id(), f.id(), c.id()]).unwrap();

        let map = sm.node_index();
        // Distinct nodes, indexed by first appearance in the connectivity.
        assert_eq!(map.len(), 6);
        assert_eq!(map[&a.id()], 0);
        assert_eq!(map[&b.id()], 1);
        assert_eq!(map[&c.id()], 2);
        assert_eq!(map[&d.id()], 3);
        assert_eq!(map[&e.id()], 4);
        assert_eq!(map[&f.id()], 5);
        // Cached: a second call returns the same populated map.
        assert_eq!(sm.node_index().len(), 6);
    }

    #[test]
    fn sealed_submesh_refuses_add_cell() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        assert!(!sm.is_sealed());
        sm.seal();
        assert!(sm.is_sealed());
        // Both mutating paths are now blocked with MeshSealed.
        assert!(matches!(
            sm.add_cell(&[b.id()]).unwrap_err(),
            PyrucastError::MeshSealed
        ));
        assert!(matches!(
            sm.add_cell_taking(&[b.id()]).unwrap_err(),
            PyrucastError::MeshSealed
        ));
        assert_eq!(sm.cell_count(), 1);
        // The refused cell left no lingering incref on b.
        assert_eq!(coords.read().refcount(b.id()), 1);
    }

    #[test]
    fn seal_via_handle_and_is_idempotent() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let h = Handle::new({
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm
        });
        seal(&h).unwrap();
        seal(&h).unwrap(); // idempotent
        assert!(h.read().is_sealed());
        assert!(matches!(
            h.write().add_cell(&[a.id()]).unwrap_err(),
            PyrucastError::MeshSealed
        ));
    }

    #[test]
    fn duplicate_is_unsealed_and_reincrefs() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        sm.seal();

        let mut copy = sm.duplicate().unwrap();
        // Copy carries the same connectivity but is not sealed.
        assert!(!copy.is_sealed());
        assert_eq!(copy.connectivity(), sm.connectivity());
        // Each node is now referenced by the original AND the copy (+ Node).
        {
            let cf = coords.read();
            assert_eq!(cf.refcount(a.id()), 3);
        }
        // The copy is editable even though the source is frozen.
        let d = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        copy.add_cell(&[a.id(), b.id(), d.id()]).unwrap();
        assert_eq!(copy.cell_count(), 2);
        assert_eq!(sm.cell_count(), 1);
    }

    #[test]
    fn remap_nodes_rewrites_connectivity_and_moves_refcounts() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b2.id(), c.id()]).unwrap();
        sm.add_cell(&[b2.id(), c.id(), a.id()]).unwrap();

        // b2 appears twice in the connectivity: two units, plus its Node.
        assert_eq!(coords.read().refcount(b2.id()), 3);
        assert_eq!(coords.read().refcount(b.id()), 1);

        let map = HashMap::from([(b2.id(), b.id()), (a.id(), a.id())]);
        assert_eq!(sm.remap_nodes(&map).unwrap(), 2, "two slots rewritten");

        assert_eq!(sm.cell_count(), 2, "renaming never changes the cells");
        assert_eq!(sm.connectivity()[1], b.id());
        assert_eq!(sm.connectivity()[3], b.id());
        // The two units moved from b2 to b; the identity entry moved nothing.
        assert_eq!(coords.read().refcount(b2.id()), 1);
        assert_eq!(coords.read().refcount(b.id()), 3);

        // Re-applying the same map is a no-op (idempotent by construction).
        assert_eq!(sm.remap_nodes(&map).unwrap(), 0);
    }

    #[test]
    fn remap_nodes_drops_the_derived_caches() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        sm.add_cell(&[a.id(), b2.id()]).unwrap();
        // Populate the node_index cache before the rename.
        assert!(sm.node_index().contains_key(&b2.id()));

        sm.remap_nodes(&HashMap::from([(b2.id(), b.id())])).unwrap();
        let index = sm.node_index();
        assert!(
            index.contains_key(&b.id()),
            "cache rebuilt on the new nodes"
        );
        assert!(!index.contains_key(&b2.id()));
    }

    #[test]
    fn remap_nodes_refuses_a_sealed_submesh() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let b2 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        sm.add_cell(&[a.id(), b2.id()]).unwrap();
        sm.seal();

        assert!(matches!(
            sm.remap_nodes(&HashMap::from([(b2.id(), b.id())]))
                .unwrap_err(),
            PyrucastError::MeshSealed
        ));
        assert_eq!(sm.connectivity()[1], b2.id(), "left untouched");
    }

    #[test]
    fn mesh_duplicate_yields_editable_copy() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        sm.add_cell(&[a.id()]).unwrap();
        sm.seal();
        let mesh = Mesh::from_submesh(sm);

        let mut copy = mesh.duplicate().unwrap();
        assert_eq!(copy.cell_count().unwrap(), 1);
        // A fresh submesh handle: editable.
        copy.add_cell(&[b.id()]).unwrap();
        assert_eq!(copy.cell_count().unwrap(), 2);
    }

    #[test]
    fn submesh_add_cell_invalid_arity() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        let err = sm.add_cell(&[a.id()]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
        // No increment should have survived the failure.
        assert_eq!(coords.read().refcount(a.id()), 1);
    }

    #[test]
    fn submesh_add_cell_collected_node_rollback() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let dead_id = coords.write().add_node(&[2.0]).unwrap();
        // dead_id starts at refcount=1; decrement then collect.
        {
            let mut c = coords.write();
            c.decref(dead_id).unwrap();
            assert_eq!(c.gc(), 1);
        }

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        // a (live), b (live), dead_id (collected) → add_cell fails after
        // increfing a and b. The rollback must undo those increfs.
        let err = sm.add_cell(&[a.id(), b.id(), dead_id]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
        {
            let cf = coords.read();
            assert_eq!(cf.refcount(a.id()), 1, "a must be rolled back");
            assert_eq!(cf.refcount(b.id()), 1, "b must be rolled back");
        }
        assert_eq!(sm.cell_count(), 0);
    }

    #[test]
    fn mesh_aggregates_submeshes_same_config() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let cc = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let sm_pts = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[a.id()]).unwrap();
            sm.add_cell(&[b.id()]).unwrap();
            Handle::new(sm)
        };
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), cc.id()]).unwrap();
            Handle::new(sm)
        };

        let mut mesh = Mesh::empty();
        mesh.add_sub(sm_pts).unwrap();
        mesh.add_sub(sm_tri).unwrap();
        assert_eq!(mesh.len(), 2);
        assert_eq!(mesh.cell_count().unwrap(), 3); // 2 points + 1 triangle
    }

    #[test]
    fn mesh_element_types_and_cell_counts() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        m.add_cell(&[b.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            Handle::new(sm)
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
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();

        let mut m = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        m.add_cell(&[a.id()]).unwrap();
        let sm_tri = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
            Handle::new(sm)
        };
        m.add_sub(sm_tri).unwrap();

        let et0 = m[0].read().element_type();
        let et1 = m[1].read().element_type();
        assert_eq!(et0, ElementType::POI1);
        assert_eq!(et1, ElementType::TRI3);

        let types: Vec<ElementType> = (&m).into_iter().map(|h| h.read().element_type()).collect();
        assert_eq!(types, vec![ElementType::POI1, ElementType::TRI3]);
    }

    #[test]
    fn mesh_node_access_by_indices() {
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let coords = Handle::new(Coords::new(1).unwrap());
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
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let s1 = Handle::new(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let s2 = Handle::new(SubMesh::poi1_from_nodes(std::slice::from_ref(&b)).unwrap());

        // sub | sub → Mesh
        let m = Mesh::union_subs(&s1, &s2).unwrap();
        assert_eq!(m.len(), 2);

        // aggregate | sub → Mesh
        let s3 = Handle::new(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let m2 = m.union_sub(&s3).unwrap();
        assert_eq!(m2.len(), 3);
    }

    #[test]
    fn node_union_node_and_mesh_union_node() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();

        let m = a.union(&b).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m.unit().unwrap().read().cell_count(), 2);

        let m2 = m.union_node(&c).unwrap();
        assert_eq!(m2.unit().unwrap().read().cell_count(), 3);
    }

    #[test]
    fn mesh_union_node_rejects_non_unitary_poi1() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.5, 1.0]).unwrap();
        let mut tri = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        tri.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        // Non-POI1 → error.
        assert!(tri.union_node(&a).is_err());
    }

    #[test]
    fn to_poi1_caches_companion_on_a_sealed_submesh() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let h = Handle::new(sm);

        // Unsealed: nothing memoized — a fresh cloud each call.
        let u1 = h.read().to_poi1().unwrap();
        let u2 = h.read().to_poi1().unwrap();
        assert!(!u1.same_object(&u2), "unsealed submesh is not memoized");

        // Sealed: the companion is built once and every call shares its slot.
        seal(&h).unwrap();
        let p1 = h.read().to_poi1().unwrap();
        let p2 = h.read().to_poi1().unwrap();
        assert!(
            p1.same_object(&p2),
            "sealed submesh memoizes its POI1 companion"
        );
        assert_eq!(p1.read().element_type(), ElementType::POI1);
        assert_eq!(p1.read().cell_count(), 3); // three distinct nodes
    }
}

#[cfg(test)]
mod nearest_node_tests {
    use super::*;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    #[test]
    fn nearest_on_grid() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n00 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n10 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n11 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[n00.id(), n10.id(), n11.id()]).unwrap();
        let mesh = Mesh::from_submesh(sm);

        // Closest to a point just past the far corner is n11.
        let found = mesh.nearest_node(&[0.9, 0.9]).unwrap();
        assert_eq!(found.id(), n11.id());

        // Closest to the origin is n00.
        let found = mesh.nearest_node(&[-0.2, 0.1]).unwrap();
        assert_eq!(found.id(), n00.id());
    }

    #[test]
    fn dimension_mismatch_errors() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let mut sm = SubMesh::new(coords, ElementType::SEG2);
        sm.add_cell(&[a.id(), b.id()]).unwrap();
        let mesh = Mesh::from_submesh(sm);
        assert!(mesh.nearest_node(&[0.0, 0.0, 0.0]).is_err());
    }
}
