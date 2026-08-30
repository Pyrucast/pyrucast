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
use serde::{Deserialize, Serialize};
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// // Une zone homogène : un type d'élément, un repère, une connectivité
/// // à plat. Elle se fige dès qu'un consommateur la capture.
/// assert_eq!(sm.element_type(), ElementType::TRI3);
/// assert_eq!(sm.cell_count(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    /// Once **sealed**, the connectivity is frozen: [`SubMesh::add_cell`] and
    /// [`SubMesh::add_cell_taking`] refuse to run. A submesh is sealed the
    /// first time a non-mesh consumer (finite-element space, field, matrix, …)
    /// captures its handle, so those consumers can never be left referencing
    /// stale cells. The seal is permanent for the object's lifetime.
    /// `serde(default)` keeps older snapshots (without the field) readable.
    #[serde(default)]
    sealed: bool,
    /// Lazily-built `NodeId → index` map over the **distinct** nodes of the
    /// connectivity, in first-appearance order. Consumers that need a node
    /// lookup (node fields, …) read it in place while holding their store
    /// guard on this submesh — no copy — so the O(n) build is paid once and
    /// mutualised across every field on this support. Not serialized — it is
    /// derived from `connectivity` and rebuilt on demand after a reload.
    /// Dropped by every connectivity mutation (see `invalidate_caches`), so it
    /// can never go stale.
    #[serde(skip)]
    node_index: OnceLock<HashMap<NodeId, usize>>,
    /// Lazily-built **canonical POI1 companion**: the node cloud of this
    /// submesh's distinct nodes, materialised once and shared. Every consumer
    /// that projects this submesh to its nodes ([`SubMesh::to_poi1`]) gets the
    /// *same* store slot, so their node fields pair under
    /// [`same_support`](crate::containers::field::SubField::same_support) — a
    /// stiffness block's support, a `restrict` onto this mesh, and a
    /// `divergence`/`flux` output over it all land on one handle and combine
    /// directly. Not serialized (derived from `connectivity`, rebuilt on
    /// demand). Dropped by every connectivity mutation (see
    /// `invalidate_caches`): the companion answers for **one** state of this
    /// submesh, and the fields already sitting on it keep it alive on their
    /// own — they stay valid, on that earlier node cloud.
    #[serde(skip)]
    poi1_companion: OnceLock<Handle<SubMesh>>,
}

impl SubMesh {
    /// Create an empty submesh for the given element type, attached to `coords`.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// // Une zone est **homogène** : un seul type d'élément, un seul repère.
    /// let mut z = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// z.add_cell(&[n[0].id(), n[1].id()])?;
    /// assert_eq!(z.cell_count(), 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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

    /// Build a submesh from a **whole flat connectivity** at once: cell `i`
    /// occupies `connectivity[i*npc..(i+1)*npc]`, so its length must be a
    /// multiple of the element type's node count.
    ///
    /// The bulk twin of [`add_cell`](SubMesh::add_cell), and the seam every
    /// operator that produces a big mesh should sit on. `add_cell` takes the
    /// `Coords` write lock and drops the derived caches **once per cell**;
    /// on a million cells that, not the connectivity itself, is the cost. Here
    /// the nodes are validated and increfed in a single locked pass — one unit
    /// per occurrence, as always — and the caches are simply never built.
    ///
    /// Nothing is increfed unless every id names a live node, so a rejected
    /// call leaves the `Coords` exactly as it was.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::SubMesh;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// // Deux triangles posés d'un seul coup, connectivité à plat.
    /// let ids: Vec<_> = n.iter().map(|x| x.id()).collect();
    /// let sm = SubMesh::from_connectivity(
    ///     coords.clone(),
    ///     ElementType::TRI3,
    ///     vec![ids[0], ids[1], ids[2], ids[1], ids[3], ids[2]],
    /// )?;
    /// assert_eq!(sm.cell_count(), 2);
    /// // Une unité par occurrence : le nœud 2 sert deux fois, plus son `Node`.
    /// assert_eq!(coords.read().refcount(ids[2]), 3);
    /// // Une longueur qui ne tombe pas juste sur le type d'élément est refusée.
    /// assert!(SubMesh::from_connectivity(coords.clone(), ElementType::TRI3, vec![ids[0]]).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn from_connectivity(
        coords: Handle<Coords>,
        element_type: ElementType,
        connectivity: Vec<NodeId>,
    ) -> Result<Self> {
        let npc = element_type.nodes_per_cell();
        if npc == 0 || !connectivity.len().is_multiple_of(npc) {
            return Err(PyrucastError::Message(format!(
                "from_connectivity({element_type}): {} node(s) is not a multiple of {npc}",
                connectivity.len()
            )));
        }
        coords.write().incref_all(&connectivity)?;
        Ok(Self {
            element_type,
            coords,
            connectivity,
            face_color: RgbColor::default(),
            sealed: false,
            node_index: OnceLock::new(),
            poi1_companion: OnceLock::new(),
        })
    }

    /// Whether this submesh is sealed (connectivity frozen).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// let mesh = Mesh::from_submesh(sm);
    /// assert!(!mesh.get(0)?.read().is_sealed());
    /// // Le premier consommateur non-maillage scelle la zone.
    /// let _fes = FiniteElementSpace::lagrange1(&mesh)?;
    /// assert!(mesh.get(0)?.read().is_sealed());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// `NodeId → index` map over the **distinct** nodes of the connectivity,
    /// in first-appearance order (the order [`SubNodeField`](crate::containers::node_field::SubNodeField) snapshots its
    /// support in). Built once and cached; callers keep their read guard on
    /// this submesh while using the returned reference — no copy.
    ///
    /// The map is derived from `connectivity`, and every mutator drops it, so
    /// it can never go stale. (It is queried through a sealed support in
    /// practice — a field's, which can no longer move at all.)
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// // `NodeId → rang`, sur les nœuds **distincts**, en ordre de première
    /// // apparition : c'est ce qui donne aux champs leur adressage.
    /// sm.seal();
    /// assert_eq!(sm.node_index()[&n[1].id()], 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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

    /// Drop the caches derived from the connectivity — the `NodeId → index`
    /// map and the POI1 companion. Called by every mutator, right after the
    /// connectivity moves.
    ///
    /// Dropping the companion handle does **not** invalidate the node fields
    /// built on it: they hold it themselves, so it stays alive, sealed, with
    /// its node refcounts. They are simply defined on the node cloud of the
    /// submesh as it was; the next [`SubMesh::to_poi1`] materialises a fresh
    /// companion for the new state, and
    /// [`restrict`](fn@crate::ops::node_field::restrict) carries a field from
    /// one to the other.
    fn invalidate_caches(&mut self) {
        self.node_index = OnceLock::new();
        self.poi1_companion = OnceLock::new();
    }

    /// Seal this submesh: freeze its connectivity permanently. After this,
    /// [`SubMesh::add_cell`] / [`SubMesh::add_cell_taking`] return
    /// [`PyrucastError::MeshSealed`]. Idempotent.
    ///
    /// Called by the container layer whenever a non-mesh object captures the
    /// submesh — see the free function [`seal`]. A bare [`Mesh`] holding the
    /// submesh does **not** seal it (a mesh may keep growing until consumed).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// // Sceller à la main : la connectivité gèle, et pour de bon.
    /// sm.seal();
    /// assert!(sm.is_sealed());
    /// assert!(sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).is_err());
    /// sm.seal(); // idempotent
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn seal(&mut self) {
        self.sealed = true;
    }

    /// Face colour used when this submesh is drawn (no numerical effect).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # use pyrucast::atoms::RgbColor;
    /// // Une couleur d'affichage, sans effet numérique.
    /// sm.set_face_color(RgbColor::new(220, 60, 60));
    /// assert_eq!(sm.face_color().r, 220);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn face_color(&self) -> RgbColor {
        self.face_color
    }

    /// Replace the face colour used when this submesh is drawn.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # use pyrucast::atoms::RgbColor;
    /// let rouge = RgbColor::new(220, 60, 60);
    /// sm.set_face_color(rouge);
    /// assert_eq!(sm.face_color(), rouge);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn set_face_color(&mut self, color: RgbColor) {
        self.face_color = color;
    }

    /// Add a cell. The length of `nodes` must equal
    /// `element_type.nodes_per_cell()`, and each node must be alive in the
    /// `Coords`; each node is increfed. On increment failure
    /// (invalid / collected id), the increfs already performed for this
    /// cell are rolled back.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// // Raccourci du cas unitaire : ajoute dans la zone unique.
    /// assert_eq!(mesh.cell_count().unwrap(), 1);
    /// ```
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
        // After the `Coords` guard above is released: dropping the previous
        // companion decrefs its nodes, which takes that same lock.
        self.invalidate_caches();
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// // La variante qui **prend** la propriété au lieu de l'acquérir : pour
    /// // un nœud tout juste créé, dont le compteur vaut déjà 1.
    /// let id = coords.write().add_node(&[2.0, 2.0])?;
    /// let mut poi = SubMesh::new(coords.clone(), ElementType::POI1);
    /// poi.add_cell_taking(&[id])?;
    /// assert_eq!(poi.cell_count(), 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
        self.invalidate_caches();
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// // L'échappatoire au sceau : une copie indépendante, jamais scellée,
    /// // partageant les **mêmes** nœuds (leurs compteurs seuls montent).
    /// sm.seal();
    /// let mut copie = sm.duplicate()?;
    /// assert!(!copie.is_sealed());
    /// copie.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// assert_eq!(copie.cell_count(), 2);
    /// assert_eq!(sm.cell_count(), 1); // l'original n'a pas bougé
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn duplicate(&self) -> Result<SubMesh> {
        // One locked pass for the whole connectivity, increfs included.
        let mut copy = SubMesh::from_connectivity(
            self.coords.clone(),
            self.element_type,
            self.connectivity.clone(),
        )?;
        copy.face_color = self.face_color;
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # use std::collections::HashMap;
    /// // Réécrire **à quel nœud** une maille se réfère — les positions ne
    /// // bougent pas. Refusé sur une zone scellée : sa numérotation est tenue.
    /// let autre = Node::create_in(coords.clone(), &[5.0, 5.0])?;
    /// let map = HashMap::from([(n[2].id(), autre.id())]);
    /// assert_eq!(sm.remap_nodes(&map)?, 1);
    /// sm.seal();
    /// assert!(sm.remap_nodes(&map).is_err());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn remap_nodes(&mut self, map: &HashMap<NodeId, NodeId>) -> Result<usize> {
        self.remap_with(|old| map.get(&old).copied().unwrap_or(old))
    }

    /// The rename [`merge_nodes`](fn@crate::ops::mesh::merge_nodes) welds
    /// with: the same contract as [`remap_nodes`](SubMesh::remap_nodes), read
    /// off a **dense table** indexed by `NodeId.0` (`table[i] == i` for a node
    /// that stays) instead of a `HashMap`. On a ten-million-cell weld the map
    /// alone is the cost — `NodeId` is already an index into the `Coords`.
    ///
    /// The table must cover the whole `Coords` this submesh sits in; indexing
    /// it is the caller's contract, as it is for
    /// [`Coords::position_alive`](crate::coords::Coords).
    pub(crate) fn remap_nodes_dense(&mut self, table: &[u32]) -> Result<usize> {
        self.remap_with(|old| NodeId(table[old.0 as usize]))
    }

    /// The shared body of the two renames: rewrite every slot whose image
    /// differs from it, moving one refcount unit per rewritten slot.
    ///
    /// One `Coords` lock for the whole submesh, and no list of changes built
    /// on the side: the images are validated first, inside that same lock, so
    /// the increments that follow cannot fail and the run stays
    /// all-or-nothing without a rollback path.
    fn remap_with(&mut self, image: impl Fn(NodeId) -> NodeId) -> Result<usize> {
        if self.sealed {
            return Err(PyrucastError::MeshSealed);
        }

        let mut rewritten = 0;
        {
            let mut c = self.coords.write();
            // Nothing is written until every image is known to be live.
            for &old in &self.connectivity {
                let new = image(old);
                if new != old && !c.is_alive(new) {
                    return Err(PyrucastError::Message(format!(
                        "remap_nodes: node {} is not found or collected",
                        new.0
                    )));
                }
            }
            for slot in &mut self.connectivity {
                let new = image(*slot);
                if new != *slot {
                    c.incref(new)?;
                    c.decref(*slot)?;
                    *slot = new;
                    rewritten += 1;
                }
            }
        }

        if rewritten > 0 {
            // Both caches are derived from the connectivity that just moved.
            self.invalidate_caches();
        }
        Ok(rewritten)
    }

    /// Element type of the submesh.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// assert_eq!(sm.element_type(), ElementType::TRI3);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Number of cells in the submesh.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// assert_eq!(sm.cell_count(), 1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn cell_count(&self) -> usize {
        self.connectivity.len() / self.element_type.nodes_per_cell()
    }

    /// Flat connectivity buffer (all cells concatenated).
    pub(crate) fn connectivity(&self) -> &[NodeId] {
        &self.connectivity
    }

    /// Handle to the owning `Coords` (internal clone).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # use pyrucast::handle::Handle as H;
    /// assert!(H::same_object(&sm.coords(), &coords)); // partagé, pas copié
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn coords(&self) -> Handle<Coords> {
        self.coords.clone()
    }

    /// Build a POI1 submesh with **one cell per [`Node`]**, in the given
    /// order. The [`Coords`] is taken from the nodes themselves
    /// (every [`Node`] carries its own — project convention). Errors if
    /// `nodes` is empty (no Coords to attach to).
    ///
    /// Lower-level form when you already hold the ids and the coords:
    /// [`SubMesh::poi1_from_node_ids`]. The canonical, parent-level form is
    /// the operator
    /// [`ops::mesh::poi1_from_nodes`](crate::ops::mesh::poi1_from_nodes()).
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// // Un nuage de points : une maille POI1 par nœud, dans l'ordre donné.
    /// // La forme canonique est l'opérateur `ops::mesh::poi1_from_nodes`.
    /// let nuage = SubMesh::poi1_from_nodes(&n)?;
    /// assert_eq!(nuage.cell_count(), 3);
    /// assert!(SubMesh::poi1_from_nodes(&[]).is_err()); // aucun repère où s'attacher
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// // La forme basse quand on tient déjà les identifiants et le repère.
    /// let ids: Vec<_> = n.iter().map(|x| x.id()).collect();
    /// let nuage = SubMesh::poi1_from_node_ids(coords.clone(), &ids)?;
    /// assert_eq!(nuage.element_type(), ElementType::POI1);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
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
    /// **Cached, per submesh.** The companion is built at most once per state
    /// of the connectivity, and every later call returns the *same* store slot,
    /// so all the node fields that project this submesh to its nodes pair under
    /// [`same_support`](crate::containers::field::SubField::same_support): a
    /// stiffness block's support (built this way in every physics' `new`), a
    /// [`restrict`](fn@crate::ops::node_field::restrict) onto this mesh, and a
    /// `divergence`/`flux`/`internal_forces` output over it all share one handle
    /// and combine directly by the field operators. This is what lets
    /// `solve(K, f) - restrict(g, mesh)` and `&K * &restrict(f, mesh)` line up.
    ///
    /// The cache does **not** require `self` to be sealed: every mutator drops
    /// it (see `invalidate_caches`), so a submesh that keeps growing simply
    /// hands out a new companion afterwards. The companion itself is always
    /// sealed — it is what node fields index their rows by. Shared building
    /// block: [`crate::ops::mesh::to_poi1()`] applies it submesh-by-submesh.
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
        // Memoize. On a race the loser drops its build and everyone reads the
        // winner's slot. Mutating `self` later drops this slot, so what is
        // cached always answers for the current connectivity.
        let _ = self.poi1_companion.set(handle);
        Ok(self
            .poi1_companion
            .get()
            .expect("populated on this path")
            .clone())
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # use pyrucast::containers::mesh::seal;
/// // Sceller par la fonction libre plutôt que par la méthode : elle prend
/// // le verrou d'écriture **seulement si nécessaire**, ce qui lui permet
/// // de sceller un support qu'un lecteur tient déjà — un `view` de champ.
/// let zone = maillage.get(0)?;
/// assert!(!zone.read().is_sealed());
/// seal(&zone)?;
/// assert!(zone.read().is_sealed());
/// // Idempotent, et sans écriture la seconde fois : le guard de lecture
/// // ci-dessous coexiste, là où un verrou d'écriture s'interbloquerait.
/// let lecteur = zone.read();
/// seal(&zone)?;
/// assert!(lecteur.is_sealed());
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::{NodeField, SubNodeField};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// // L'agrégat de zones : chacune homogène, l'ensemble ne l'étant pas.
/// // C'est ce qui permet à un maillage de mêler triangles et quadrangles
/// // sans que rien, en aval, ait à s'en soucier.
/// assert_eq!(maillage.len(), 1);
/// assert_eq!(maillage.cell_count()?, 1);
/// assert_eq!(maillage.element_types()?, vec![ElementType::TRI3]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// // Le même `|` que celui des agrégats, appliqué à deux nœuds : un
    /// // maillage POI1 unitaire, qu'on fait ensuite croître nœud par nœud.
    /// let deux = n[0].union(&n[1])?;
    /// assert_eq!(deux.cell_count()?, 2);
    /// assert_eq!(deux.element_types()?, vec![ElementType::POI1]);
    /// assert_eq!(deux.union_node(&n[2])?.cell_count()?, 3);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn union(&self, other: &Node) -> Result<Mesh> {
        let sm = SubMesh::poi1_from_nodes(&[self.clone(), other.clone()])?;
        Ok(Mesh::from_submesh(sm))
    }
}

impl Mesh {
    /// `mesh.union_node(node)` → a unitary POI1 [`Mesh`] holding this mesh's
    /// points plus `node`. Errors unless `self` is **unitary and POI1**
    /// (exactly one POI1 submesh). Python: `mesh | node`.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// // L'opérande gauche doit être un **nuage POI1 unitaire** : c'est une
    /// // union de points, pas l'ajout d'un point à un maillage quelconque.
    /// let nuage = pyrucast::ops::mesh::poi1_from_nodes(&[n[0].clone()]).unwrap();
    /// let deux_points = nuage.union_node(&n[1]).unwrap();
    /// assert_eq!(deux_points.cell_count().unwrap(), 2);
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// assert_eq!(mesh.cell_count().unwrap(), 1); // toutes zones confondues
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # use pyrucast::handle::Handle as H;
    /// // Le maillage ne porte pas la `Coords` : il la retrouve par ses zones.
    /// assert!(H::same_object(&mesh.coords().unwrap(), &coords));
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::atoms::ElementType;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # use pyrucast::aggregate::Aggregate;
    /// // Le raccourci du cas à une seule zone.
    /// let mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
    /// assert_eq!(mesh.len(), 1);
    /// ```
    pub fn from_submesh(sub: SubMesh) -> Self {
        let mut mesh = Self::default();
        mesh.subs.push(Handle::new(sub));
        mesh
    }

    /// Deep-copy the whole mesh: every submesh is [`SubMesh::duplicate`]d
    /// into a fresh, unsealed submesh under a new handle. The copy is fully
    /// editable even when the source's submeshes have been sealed by their
    /// consumers; nodes are shared (same `Coords`), only their refcounts grow.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
    /// FiniteElementSpace::lagrange1(&mesh).unwrap(); // scelle la zone
    /// let mut copie = mesh.duplicate().unwrap(); // neuve, modifiable
    /// copie.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// assert_eq!(copie.cell_count().unwrap(), 2);
    /// ```
    pub fn duplicate(&self) -> Result<Mesh> {
        let mut copy = Self::default();
        for sm in self {
            let dup = sm.read().duplicate()?;
            copy.subs.push(Handle::new(dup));
        }
        Ok(copy)
    }

    /// Paint **every** submesh with the same face colour, and hand the mesh
    /// back so the call chains — `mesh::circle(…)?.set_face_color(rouge)`.
    ///
    /// What comes back holds the very **same** zones (the same handles), not
    /// copies of them: it *is* this mesh, and the two are interchangeable.
    /// The colour is viz metadata — [`SubMesh::set_face_color`] on each zone —
    /// so a **sealed** mesh takes it without complaint; the seal freezes the
    /// connectivity, not the way it is drawn.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node, RgbColor};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let seg = { let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    /// #     sm.add_cell(&[n[0].id(), n[1].id()]).unwrap(); Handle::new(sm) };
    /// # mesh.add_sub(seg).unwrap();
    /// // Une couleur pour toutes les zones, sans boucle — et le maillage
    /// // revient, pour enchaîner.
    /// let rouge = RgbColor::new(220, 60, 60);
    /// assert_eq!(mesh.set_face_color(rouge).cell_count()?, 2);
    /// assert!(mesh.iter().all(|z| z.read().face_color() == rouge));
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn set_face_color(&self, color: RgbColor) -> Mesh {
        let mut same = Self::default();
        for sub in self {
            sub.write().set_face_color(color);
            same.subs.push(sub.clone());
        }
        same
    }

    /// Add a cell directly when the mesh has exactly one submesh.
    ///
    /// ```
    /// # use pyrucast::aggregate::Aggregate;
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::field::SubField;
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::containers::node_field::{NodeField, SubNodeField};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # use pyrucast::ops::mesh;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
    /// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// # let maillage = Mesh::from_submesh(sm);
    /// // Le raccourci du cas **unitaire** : il ajoute dans la zone unique, et
    /// // refuse dès qu'il y en a plusieurs — l'ambiguïté serait silencieuse.
    /// let mut m = maillage;
    /// m.add_cell(&[n[0].id(), n[1].id(), n[2].id()])?;
    /// assert_eq!(m.cell_count()?, 2);
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn add_cell(&mut self, nodes: &[NodeId]) -> Result<usize> {
        if self.len() != 1 {
            return Err(PyrucastError::Message(
                "add_cell: mesh must have exactly one submesh".into(),
            ));
        }
        self.subs[0].write().add_cell(nodes)
    }

    /// Element type of each submesh, in order.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// assert_eq!(mesh.element_types().unwrap(), vec![ElementType::TRI3]);
    /// ```
    pub fn element_types(&self) -> Result<Vec<ElementType>> {
        self.iter().map(|sm| Ok(sm.read().element_type())).collect()
    }

    /// Cell count of each submesh, in order.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// assert_eq!(mesh.cell_counts().unwrap(), vec![1]); // une entrée par zone
    /// ```
    pub fn cell_counts(&self) -> Result<Vec<usize>> {
        self.iter().map(|sm| Ok(sm.read().cell_count())).collect()
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
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// // Le nœud le plus proche d'un point quelconque, distance euclidienne.
    /// let proche = mesh.nearest_node(&[0.9, 0.1]).unwrap();
    /// assert_eq!(proche.position().unwrap(), vec![1.0, 0.0]);
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// let a = mesh.node(0, 0, 0).unwrap(); // zone 0, cellule 0, sommet 0
    /// assert_eq!(a.position().unwrap(), vec![0.0, 0.0]);
    /// ```
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
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// let c = mesh.cell(0, 0).unwrap();
    /// assert_eq!(c.nodes_per_cell().unwrap(), 3);
    /// ```
    pub fn cell(&self, submesh_idx: usize, cell_idx: usize) -> Result<Cell> {
        let sm = self.get(submesh_idx)?;
        Cell::new(sm, cell_idx)
    }

    /// Iterator over every cell of submesh `submesh_idx`.
    ///
    /// ```
    /// # use pyrucast::atoms::{ElementType, Node};
    /// # use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// # let coords = Handle::new(Coords::new(2).unwrap());
    /// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    /// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
    /// # let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
    /// # mesh.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
    /// assert_eq!(mesh.cells(0).unwrap().count(), 1);
    /// ```
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

// ─── Archive ────────────────────────────────────────────────────────────────

impl crate::archive::Archivable for SubMesh {
    const TAG: &'static str = "SubMesh";

    /// Re-increment the nodes this submesh uses.
    ///
    /// `Drop` decrements them, so without this the counts would go negative the
    /// day the reloaded mesh dies. The owning `Coords` has already been decoded
    /// and zeroed — that is what the post-order is for.
    fn on_load(&mut self) {
        let mut coords = self.coords.write();
        for &n in &self.connectivity {
            let _ = coords.incref(n);
        }
    }
}

impl crate::archive::Archivable for Mesh {
    const TAG: &'static str = "Mesh";
}

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
    fn from_connectivity_refuses_a_dead_node_without_increfing_anything() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let dead_id = coords.write().add_node(&[2.0]).unwrap();
        {
            let mut c = coords.write();
            c.decref(dead_id).unwrap();
            assert_eq!(c.gc(), 1);
        }

        let err = SubMesh::from_connectivity(
            coords.clone(),
            ElementType::TRI3,
            vec![a.id(), b.id(), dead_id],
        )
        .unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
        // Validation comes first: nothing was increfed on the way out.
        let cf = coords.read();
        assert_eq!((cf.refcount(a.id()), cf.refcount(b.id())), (1, 1));
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
    fn to_poi1_caches_companion_even_unsealed() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let h = Handle::new(sm);

        // Unsealed, and still memoized: the companion answers for the current
        // connectivity, and a mutation drops it.
        let u1 = h.read().to_poi1().unwrap();
        let u2 = h.read().to_poi1().unwrap();
        assert!(
            u1.same_object(&u2),
            "companion memoized on an unsealed submesh"
        );
        assert!(
            !h.read().is_sealed(),
            "asking for the companion does not seal"
        );
        assert!(u1.read().is_sealed(), "the companion itself is sealed");
        assert_eq!(u1.read().element_type(), ElementType::POI1);
        assert_eq!(u1.read().cell_count(), 3); // three distinct nodes

        // Sealing changes nothing: same slot.
        seal(&h).unwrap();
        assert!(h.read().to_poi1().unwrap().same_object(&u1));
    }

    #[test]
    fn growing_a_submesh_drops_its_poi1_companion() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let d = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();

        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let h = Handle::new(sm);

        let before = h.read().to_poi1().unwrap();
        assert_eq!(before.read().cell_count(), 3);

        // One more cell ⇒ a different node cloud, so a different companion.
        h.write().add_cell(&[b.id(), d.id(), c.id()]).unwrap();
        let after = h.read().to_poi1().unwrap();
        assert!(
            !before.same_object(&after),
            "the stale companion was dropped"
        );
        assert_eq!(after.read().cell_count(), 4);
        // The old one is untouched — whoever holds it (a field) still reads
        // the three nodes it was built on.
        assert_eq!(before.read().cell_count(), 3);
        // And its node_index went with it.
        assert_eq!(h.read().node_index()[&d.id()], 3);
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
    fn set_face_color_paints_every_zone_even_sealed() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let seg = {
            let mut sm = SubMesh::new(coords, ElementType::SEG2);
            sm.add_cell(&[a.id(), b.id()]).unwrap();
            sm.set_face_color(RgbColor::new(1, 2, 3));
            Handle::new(sm)
        };
        mesh.add_sub(seg).unwrap();
        // Le sceau gèle la connectivité, pas la façon de la dessiner.
        crate::containers::finite_element_space::FiniteElementSpace::lagrange1(&mesh).unwrap();
        assert!(mesh.get(0).unwrap().read().is_sealed());

        let rouge = RgbColor::new(220, 60, 60);
        let retour = mesh.set_face_color(rouge);

        assert_eq!(retour.len(), 2);
        for i in 0..2 {
            // Les mêmes zones, pas des copies : le maillage rendu est celui-ci.
            assert!(retour.get(i).unwrap().same_object(&mesh.get(i).unwrap()));
            assert_eq!(mesh.get(i).unwrap().read().face_color(), rouge);
        }
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
