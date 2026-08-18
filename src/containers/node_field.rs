//! Node fields — multi-component values carried by mesh nodes.
//!
//! Hierarchy mirroring [`crate::containers::element_field`]:
//!
//! - [`SubNodeField`] — multi-component values on the nodes of **one**
//!   zone, supported by a POI1 [`SubMesh`];
//! - [`NodeField`] — aggregate of `SubNodeField`, one per zone. A node
//!   shared by several zones (an interface node) may be stored by several
//!   subs; aggregate reads take the **first** sub defining
//!   `(node, component)`, and coherence across duplicates is checked on
//!   demand by [`NodeField::check`].
//!
//! A [`SubNodeField`] stores one or more named components per node of a
//! support defined by a POI1 [`SubMesh`] (a list of nodes). The field
//! holds a `Handle<SubMesh>` on its support: the SubMesh is the
//! single owner of per-node refcounts in the [`Coords`] (its
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
//! use pyrucast::coords::Coords;
//! use pyrucast::atoms::ElementType;
//! use pyrucast::containers::mesh::SubMesh;
//! use pyrucast::atoms::Node;
//! use pyrucast::containers::node_field::SubNodeField;
//! use pyrucast::handle::Handle;
//!
//! let coords = Handle::new(Coords::new(2).unwrap());
//! let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
//! let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
//!
//! // Build a POI1 SubMesh holding [a, b], then a 2-component field on it.
//! let sm_handle = {
//!     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
//!     sm.add_cell(&[a.id()]).unwrap();
//!     sm.add_cell(&[b.id()]).unwrap();
//!     Handle::new(sm)
//! };
//!
//! let mut field = SubNodeField::from_poi1(
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
use crate::atoms::ElementType;
use crate::atoms::NodeId;
use crate::containers::field::SubField;
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Index, IndexMut};

// ─── SubNodeField ──────────────────────────────────────────────────────────────

/// Multi-component values on a snapshot of a POI1 SubMesh's node list.
///
/// Values are stored row-major: component `c` of node `i` is at index
/// `i * component_count + c` in the internal flat buffer.
#[derive(Serialize, Deserialize)]
pub struct SubNodeField {
    /// POI1 SubMesh owning the per-node refcounts. The field keeps a
    /// clone of this handle for its whole lifetime; that is the only
    /// thing keeping the support (and its nodes) alive.
    support: Handle<SubMesh>,
    /// Cached connectivity of `support` (POI1 ⇒ one node per cell).
    /// Frozen at construction by [[project-submesh-immutable-size]].
    ///
    /// **Never archived**: it is a copy of the support's connectivity, and the
    /// support is sealed the moment a field captures it, so the copy cannot
    /// drift. `on_load` takes it again — which is 4 MiB per field of a million
    /// nodes that the file does not carry.
    #[serde(skip)]
    nodes: Vec<NodeId>,
    components: Vec<String>,
    /// Row-major: `values[i * components.len() + c]`.
    values: Vec<f64>,
}

impl SubNodeField {
    /// Build a SubNodeField on the nodes of a POI1 [`SubMesh`]. The support
    /// is captured as a snapshot; subsequent changes to the SubMesh do
    /// not affect this field.
    ///
    /// Errors:
    /// - the SubMesh is not POI1,
    /// - `components` is empty,
    /// - `components` contains duplicate names.
    pub fn from_poi1(submesh: &Handle<SubMesh>, components: Vec<String>) -> Result<Self> {
        crate::containers::field::check_components("SubNodeField", &components)?;

        let nodes: Vec<NodeId> = {
            let sm = submesh.read();
            if sm.element_type() != ElementType::POI1 {
                return Err(PyrucastError::Message(format!(
                    "SubNodeField requires a POI1 SubMesh, got {}",
                    sm.element_type()
                )));
            }
            // POI1: connectivity is exactly the node list (1 node per cell).
            sm.connectivity().to_vec()
        };

        // The field snapshots this POI1 support's node list and shares its
        // handle; freeze it so the two can never diverge.
        let support = crate::containers::mesh::seal(submesh)?;

        let n_nodes = nodes.len();
        let n_comp = components.len();
        Ok(SubNodeField {
            support,
            nodes,
            components,
            values: vec![0.0; n_nodes * n_comp],
        })
    }

    /// Build a SubNodeField on the distinct nodes of **any** [`SubMesh`]. A
    /// POI1 support is shared as-is (same handle); any other element type lands
    /// on the submesh's **canonical POI1 companion** ([`SubMesh::to_poi1`]) —
    /// materialised once and cached, so every field restricted to this submesh
    /// (and the stiffness block / `divergence` / `flux` output over it) shares
    /// one support slot and pairs under `same_support`.
    pub fn from_support(submesh: &Handle<SubMesh>, components: Vec<String>) -> Result<Self> {
        let element_type = submesh.read().element_type();
        if element_type == ElementType::POI1 {
            return Self::from_poi1(submesh, components);
        }
        // Freeze the source (so later `add_cell`s cannot leave it behind, and so
        // `to_poi1` may memoize the companion), then share that cached POI1
        // support. `from_poi1` validates `components`.
        crate::containers::mesh::seal(submesh)?;
        let poi = submesh.read().to_poi1()?;
        Self::from_poi1(&poi, components)
    }

    /// Number of nodes in the support.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // `components`, `component_count`, `component_index`, … come from
    // the [`crate::containers::field::SubField`] trait.

    /// Node ids this field is defined on, in support order.
    pub(crate) fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// Handle to the owning `Coords` (derived from the support).
    pub fn coords(&self) -> Handle<Coords> {
        self.support.read().coords()
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
        let i = self
            .index_of(nid)
            .ok_or_else(|| PyrucastError::Message(format!("node {} not in field support", nid)))?;
        self.get(i, comp_idx)
    }

    /// Write a value by `(NodeId, component_index)`.
    pub fn set_by_node(&mut self, nid: NodeId, comp_idx: usize, value: f64) -> Result<()> {
        let i = self
            .index_of(nid)
            .ok_or_else(|| PyrucastError::Message(format!("node {} not in field support", nid)))?;
        self.set(i, comp_idx, value)
    }

    /// Position of a NodeId in the support, or `None` if absent.
    ///
    /// The support is a sealed POI1 SubMesh, so its `NodeId → index` map is
    /// cached and consistent with `self.nodes` (same first-appearance order).
    /// We consult that map for an O(1) lookup instead of the linear scan that
    /// used to dominate the hot write paths (`set_value`, arithmetic, …). A
    /// short read guard on the support is opened and released here; callers
    /// looping over many nodes should hoist it with [`index_of_with`].
    ///
    /// [`index_of_with`]: SubNodeField::index_of_with
    pub fn index_of(&self, nid: NodeId) -> Option<usize> {
        let support = self.support.read();
        self.index_of_with(&support, nid)
    }

    /// Position of a NodeId in the support using a caller-held read guard on
    /// the support SubMesh — the O(1) map lookup without re-locking. Meant for
    /// tight loops (per-node writes) that resolve many ids under one guard.
    pub fn index_of_with(&self, support: &SubMesh, nid: NodeId) -> Option<usize> {
        support.node_index().get(&nid).copied()
    }

    /// Build a new POI1 [`SubMesh`] whose cells are exactly the support nodes
    /// of this field, in the same order. Each node is increfed by the new
    /// submesh independently of this field's own increfs.
    ///
    /// # Example
    ///
    /// ```
    /// use pyrucast::aggregate::Aggregate;
    /// use pyrucast::coords::Coords;
    /// use pyrucast::atoms::ElementType;
    /// use pyrucast::containers::mesh::{Mesh, SubMesh};
    /// use pyrucast::atoms::Node;
    /// use pyrucast::containers::node_field::SubNodeField;
    /// use pyrucast::handle::Handle;
    ///
    /// let coords = Handle::new(Coords::new(2).unwrap());
    /// let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
    /// let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
    ///
    /// let sm_handle = {
    ///     let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    ///     sm.add_cell(&[a.id()]).unwrap();
    ///     sm.add_cell(&[b.id()]).unwrap();
    ///     Handle::new(sm)
    /// };
    ///
    /// let field = SubNodeField::from_poi1(&sm_handle, vec!["T".into()]).unwrap();
    /// let sm2 = field.support_submesh().unwrap();
    /// assert_eq!(sm2.cell_count(), 2);
    /// // Verify node order via Mesh::node (public API).
    /// let mut m = Mesh::empty();
    /// m.add_sub(Handle::new(sm2)).unwrap();
    /// assert_eq!(m.node(0, 0, 0).unwrap().id(), a.id());
    /// assert_eq!(m.node(0, 1, 0).unwrap().id(), b.id());
    /// ```
    pub fn support_submesh(&self) -> Result<SubMesh> {
        SubMesh::poi1_from_node_ids(self.coords(), &self.nodes)
    }

    /// Build a [`Mesh`] with a single POI1 submesh mirroring the support of
    /// this field.
    pub fn support_mesh(&self) -> Result<Mesh> {
        let sm_handle = Handle::new(self.support_submesh()?);
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

    // ── Helpers privés ──────────────────────────────────────────────────────

    pub(crate) fn component_value_opt(&self, nid: NodeId, comp: &str) -> Option<f64> {
        let ni = self.index_of(nid)?;
        self.component_value_at(ni, comp)
    }

    /// Value at an **already-resolved** node index and a named component, or
    /// `None` if the component is absent. Used by [`NodeFieldView`], which
    /// resolves the node index through the support's cached map.
    pub(crate) fn component_value_at(&self, node_idx: usize, comp: &str) -> Option<f64> {
        let ci = self.component_index(comp)?;
        Some(self.values[node_idx * self.components.len() + ci])
    }

    // ── Accès idiomatique ───────────────────────────────────────────────────

    /// Read a value by `(NodeId, component name)`.
    pub fn value(&self, nid: NodeId, component: &str) -> Result<f64> {
        let ni = self
            .index_of(nid)
            .ok_or_else(|| PyrucastError::Message(format!("node {} not in field support", nid)))?;
        let ci = self.component_index_or_err(component)?;
        Ok(self.values[ni * self.components.len() + ci])
    }

    /// Write a value by `(NodeId, component name)`.
    pub fn set_value(&mut self, nid: NodeId, component: &str, value: f64) -> Result<()> {
        let ni = self
            .index_of(nid)
            .ok_or_else(|| PyrucastError::Message(format!("node {} not in field support", nid)))?;
        let ci = self.component_index_or_err(component)?;
        let ncomp = self.components.len();
        self.values[ni * ncomp + ci] = value;
        Ok(())
    }

    // Scalar per-component operations (`add_to_component`, …) come from
    // the [`crate::containers::field::SubField`] trait.

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

impl crate::containers::field::SubField for SubNodeField {
    type Support = SubMesh;
    fn support(&self) -> Handle<SubMesh> {
        self.support.clone()
    }
    fn components(&self) -> &[String] {
        &self.components
    }
    fn values(&self) -> &[f64] {
        &self.values
    }
    fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
    fn same_support_with(&self, components: Vec<String>) -> Result<Self> {
        Self::from_support(&self.support, components)
    }
}

impl fmt::Debug for SubNodeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Bounded structure only — the per-node values live in `dump()`.
        f.debug_struct("SubNodeField")
            .field("support", &self.support)
            .field("node_count", &self.nodes.len())
            .field("components", &self.components)
            .finish()
    }
}

impl fmt::Display for SubNodeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SubNodeField: {} node(s), {} component(s) [{}]",
            self.nodes.len(),
            self.components.len(),
            self.components.join(", ")
        )
    }
}

impl crate::dump::Dump for SubNodeField {
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
impl Index<(NodeId, &str)> for SubNodeField {
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
impl IndexMut<(NodeId, &str)> for SubNodeField {
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

impl Clone for SubNodeField {
    fn clone(&self) -> Self {
        // Cloning the support Handle bumps the SubMesh's store refcount;
        // per-node refcounts in the Coords are already covered by
        // the shared SubMesh.
        SubNodeField {
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

crate::impl_subfield_scalar_ops!(SubNodeField);

// ─── Opérateurs field OP field (même support) ───────────────────────────────
//
// `&a + &b` (et `a + b`) délèguent à `SubField::merge_components` (union des
// composantes avec passthrough) ; faillible (même support exigé) ⇒ sortie
// `Result<SubNodeField>`.

crate::impl_subfield_field_ops!(SubNodeField);

// ─── NodeField (aggregate) ──────────────────────────────────────────────────

/// Aggregate of [`SubNodeField`] — one per zone.
///
/// Mirrors the Mesh/SubMesh and ElementField/SubElementField hierarchies:
/// a `NodeField` is a list of sub-field handles with the uniform
/// [`Aggregate`] grammar (`len`, indexing, iteration, `|` as structural
/// union). Components may differ from one zone to the next — nothing is
/// densified.
///
/// A node shared by several zones (an interface node) may be stored by
/// several subs. Aggregate reads ([`NodeField::value`]) take the **first**
/// sub defining `(node, component)`; whether the duplicates agree is
/// verified on demand by [`NodeField::check`]. Writes go through the subs
/// (`with_mut` on `field.get(i)`), exactly like `ElementField`.
#[derive(Serialize, Deserialize, Default)]
pub struct NodeField {
    subs: Vec<Handle<SubNodeField>>,
}

crate::impl_aggregate!(NodeField, SubNodeField, subfield, "subfield(s)", {
    fn check_push(&self, h: &Handle<SubNodeField>) -> Result<()> {
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

    /// Fuse zones sharing the same support `SubMesh` (union of components,
    /// shared values verified). Runs at the end of every union (`a | b`).
    /// See [`crate::ops::node_field::consolidate`](fn@crate::ops::node_field::consolidate).
    fn finalize(&mut self) -> Result<()> {
        *self = crate::ops::node_field::consolidate(self)?;
        Ok(())
    }
});
crate::impl_aggregate_dump!(NodeField);

// ─── Opérateurs NodeField OP {NodeField, f64} ───────────────────────────────
//
// `&a + &b` (zone par zone, même décomposition) via `Field::merge_field` ;
// `&a + 2.0` (diffusion scalaire) via `Field::combine_scalar`. Faillibles
// (lecture dans le store, appariement des zones) ⇒ sortie `Result<NodeField>`.

crate::impl_field_ops!(NodeField);

impl NodeField {
    /// Zero-copy view of this field's zones for read-heavy operators
    /// (`deformation`, `restrict`, viz, …). One read guard per zone **and**
    /// per zone support is held for the view's lifetime, so every
    /// `(node, component)` lookup resolves through the support's cached
    /// `NodeId → index` map in O(1). Shadows the generic
    /// [`crate::containers::field::Field::view`] to return the node-specific
    /// [`NodeFieldView`].
    pub(crate) fn view(&self) -> Result<NodeFieldView> {
        NodeFieldView::new(self)
    }

    /// One zero-initialized [`SubNodeField`] per submesh of `mesh`, all
    /// sharing the same `components`. Each sub is supported on the
    /// distinct nodes of its submesh — interface nodes shared by several
    /// submeshes are therefore stored once **per zone**.
    ///
    /// `mesh` must have at least one submesh.
    pub fn new(mesh: &Mesh, components: Vec<String>) -> Result<Self> {
        if mesh.is_empty() {
            return Err(PyrucastError::Message(
                "NodeField: mesh has no submesh".into(),
            ));
        }
        crate::containers::field::check_components("SubNodeField", &components)?;
        let mut field = Self::default();
        for h in mesh {
            let sub = SubNodeField::from_support(h, components.clone())?;
            field.add_sub(Handle::new(sub))?;
        }
        Ok(field)
    }

    /// Build a `NodeField` with an explicit `components` list per submesh.
    /// `components_per_submesh.len()` must equal `mesh.len()`.
    pub fn with(mesh: &Mesh, components_per_submesh: &[Vec<String>]) -> Result<Self> {
        if mesh.is_empty() {
            return Err(PyrucastError::Message(
                "NodeField: mesh has no submesh".into(),
            ));
        }
        if components_per_submesh.len() != mesh.len() {
            return Err(PyrucastError::Message(format!(
                "NodeField: {} component list(s) supplied for {} submesh(es)",
                components_per_submesh.len(),
                mesh.len()
            )));
        }
        let mut field = Self::default();
        for (h, comps) in mesh.iter().zip(components_per_submesh) {
            let sub = SubNodeField::from_support(h, comps.clone())?;
            field.add_sub(Handle::new(sub))?;
        }
        Ok(field)
    }

    /// Single-zone `NodeField` over the distinct nodes of one [`SubMesh`].
    pub fn from_submesh(submesh: &Handle<SubMesh>, components: Vec<String>) -> Result<Self> {
        let mut field = Self::default();
        field.add_sub(Handle::new(SubNodeField::from_support(
            submesh, components,
        )?))?;
        Ok(field)
    }

    /// Wrap a single [`SubNodeField`] into a unitary aggregate.
    pub fn from_sub(sub: SubNodeField) -> Self {
        let mut field = Self::default();
        field.subs.push(Handle::new(sub));
        field
    }

    /// Handle to the owning `Coords` (from the first sub).
    /// Errors if the aggregate is empty.
    pub fn coords(&self) -> Result<Handle<Coords>> {
        let h = self.get(0)?;
        Ok(h.read().coords())
    }

    /// Value at `(node, component)` — the **first** sub defining both
    /// wins. Errors if no sub does.
    pub fn value(&self, nid: NodeId, component: &str) -> Result<f64> {
        self.value_opt(nid, component)?.ok_or_else(|| {
            PyrucastError::Message(format!(
                "no subfield defines (node {}, component {})",
                nid, component
            ))
        })
    }

    /// Like [`NodeField::value`], but `None` when no sub defines
    /// `(node, component)`.
    pub fn value_opt(&self, nid: NodeId, component: &str) -> Result<Option<f64>> {
        for h in self {
            if let Some(v) = h.read().component_value_opt(nid, component) {
                return Ok(Some(v));
            }
        }
        Ok(None)
    }

    /// Values at `nodes` for the named `component`, returned in the **same
    /// order** as `nodes` — the batch form of [`value`](Self::value). The
    /// first sub defining each `(node, component)` pair wins; errors on the
    /// first node no sub defines. The view is built once, so this is a
    /// single pass over the zones per node.
    pub fn values_at(&self, nodes: &[NodeId], component: &str) -> Result<Vec<f64>> {
        let view = self.view()?;
        nodes
            .iter()
            .map(|&nid| view.value(nid, component))
            .collect()
    }

    /// Distinct node ids across the subs, first-seen order.
    pub fn node_ids(&self) -> Result<Vec<NodeId>> {
        let mut out: Vec<NodeId> = Vec::new();
        for h in self {
            let s = h.read();
            for &nid in s.nodes() {
                if !out.contains(&nid) {
                    out.push(nid);
                }
            }
        }
        Ok(out)
    }

    /// Number of distinct nodes across the subs.
    pub fn node_count(&self) -> Result<usize> {
        Ok(self.node_ids()?.len())
    }

    /// Visualize this field alone, as a **coloured point cloud** over
    /// its support nodes — the POI1 support carries no connectivity, so
    /// no surface can be drawn; use `Mesh::plot_with_field` with the
    /// original mesh for surfaces. `component = None` selects the first
    /// component.
    #[cfg(feature = "viz")]
    pub fn plot(
        &self,
        view: Option<crate::viz::View>,
        save: Option<&std::path::Path>,
        component: Option<&str>,
        scale: crate::viz::ColorScale,
        title: Option<&str>,
    ) -> Result<()> {
        crate::viz::render_node_field_points(self, component, scale, view, save, title)
    }

    /// Verify zone coherence: every `(node, component)` stored by several
    /// subs must hold the **same** value (exact comparison) everywhere.
    ///
    /// Reads tolerate divergence (first sub wins); this is the on-demand
    /// verification — call it before trusting a field assembled from
    /// independently mutated zones. `ops::node_field::consolidate` runs the
    /// same verification while deduplicating.
    pub fn check(&self) -> Result<()> {
        use crate::containers::field::SubField;
        use std::collections::HashMap;
        let mut seen: HashMap<(NodeId, String), f64> = HashMap::new();
        for h in self {
            let s = h.read();
            let (nodes, comps, values) = (s.nodes(), SubField::components(&*s), s.values());
            let ncomp = comps.len();
            for (ni, &nid) in nodes.iter().enumerate() {
                for (ci, comp) in comps.iter().enumerate() {
                    let v = values[ni * ncomp + ci];
                    match seen.entry((nid, comp.clone())) {
                        std::collections::hash_map::Entry::Occupied(e) => {
                            if *e.get() != v {
                                return Err(PyrucastError::Message(format!(
                                    "incoherent NodeField: node {}, component {}: {} ≠ {}",
                                    nid,
                                    comp,
                                    e.get(),
                                    v
                                )));
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(v);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Read the values at `dofs` (`(node, component)` pairs) into a dense
    /// vector, in `dofs` order. Aggregate resolution: the first zone defining a
    /// pair wins. A DOF no zone defines reads as `0.0` — the natural neutral for
    /// a right-hand side or a multiplied vector (see [`crate::ops::solver::lu::solve`]
    /// and [`Matrix::mul_field`](crate::containers::matrix::Matrix::mul_field)).
    pub fn gather(&self, dofs: &[(NodeId, String)]) -> Result<Vec<f64>> {
        let view = self.view()?;
        Ok(dofs
            .iter()
            .map(|(nid, name)| view.value_opt(*nid, name).unwrap_or(0.0))
            .collect())
    }

    /// Build a fresh single-zone `NodeField` on `coords` holding `values` at
    /// `dofs` (`(node, component)` pairs, in `values` order). The support is a
    /// POI1 submesh over the distinct nodes (first-seen order); the components
    /// are the distinct field names (first-seen order). `dofs` and `values` must
    /// have equal length; a repeated DOF keeps the last value written.
    ///
    /// The inverse of [`gather`](Self::gather): together they bridge the
    /// abstract `NodeField` and the flat DOF vectors the linear algebra speaks.
    pub fn from_dof_values(
        coords: Handle<Coords>,
        dofs: &[(NodeId, String)],
        values: &[f64],
    ) -> Result<Self> {
        if dofs.len() != values.len() {
            return Err(PyrucastError::Message(format!(
                "from_dof_values: {} dof(s) but {} value(s)",
                dofs.len(),
                values.len()
            )));
        }
        // Distinct nodes and components, first-seen order. O(n) dedup via
        // seen-sets (the previous `Vec::contains` was O(n²), a profiler hotspot
        // on the `solve` output reprojected every Newton iteration).
        let mut seen_nodes: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        let mut seen_components: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut unique_nodes: Vec<NodeId> = Vec::new();
        let mut unique_components: Vec<String> = Vec::new();
        for (nid, name) in dofs {
            if seen_nodes.insert(*nid) {
                unique_nodes.push(*nid);
            }
            if seen_components.insert(name.as_str()) {
                unique_components.push(name.clone());
            }
        }
        // POI1 support over the distinct nodes; the submesh and field both land
        // in the store and cascade-decref their nodes on drop.
        let sm_h = Handle::new(SubMesh::poi1_from_node_ids(coords, &unique_nodes)?);
        let mut sub = SubNodeField::from_poi1(&sm_h, unique_components)?;
        // Resolve component indices once and hoist one read guard on the
        // support over the whole write loop (O(1) node lookups, no re-locking).
        let comp_idx: std::collections::HashMap<&str, usize> = sub
            .components
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();
        {
            let support = sub.support.read();
            let ncomp = sub.components.len();
            for ((nid, name), &v) in dofs.iter().zip(values) {
                let ni = *support
                    .node_index()
                    .get(nid)
                    .expect("node just inserted into support");
                let ci = comp_idx[name.as_str()];
                sub.values[ni * ncomp + ci] = v;
            }
        }
        Ok(Self::from_sub(sub))
    }
}

/// Zero-copy view of a [`NodeField`]'s zones, specialised to node fields
/// (built by [`NodeField::view`]). Reads mirror the aggregate: first zone
/// defining `(node, component)` wins.
///
/// On top of the generic [`FieldView`] zone guards, this holds **one read
/// guard on each zone's support [`SubMesh`]** for the whole lifetime of the
/// view. The guard keeps the support's `NodeId → index` map alive and lets
/// every `(node, component)` lookup hit it in O(1) — no per-access lock, no
/// snapshot copy. A view is short-lived (one operator call), so holding a
/// long shared read lock on the supports is cheaper than repeatedly locking
/// or duplicating the map.
pub(crate) struct NodeFieldView {
    inner: crate::containers::field::FieldView<SubNodeField>,
    /// One shared read guard per zone, aligned with `inner.zones`, over the
    /// zone's support `SubMesh`. Held for the view's lifetime.
    supports: Vec<crate::handle::ReadGuard<SubMesh>>,
}

impl NodeFieldView {
    /// Build the view: read every zone in place, and open a read guard on
    /// each zone's support so its node-index map can be queried lock-free.
    pub(crate) fn new(field: &NodeField) -> Result<Self> {
        let inner = crate::containers::field::Field::view(field)?;
        let supports = inner.zones.iter().map(|z| z.support.read()).collect();
        Ok(Self { inner, supports })
    }

    /// Union of the zones' component names, first-seen order.
    // Consumed by the viz layer (feature-gated).
    #[allow(dead_code)]
    pub(crate) fn components(&self) -> &[String] {
        self.inner.components()
    }

    /// Value at `(node, component)` — first zone wins; errors if absent.
    pub(crate) fn value(&self, nid: NodeId, component: &str) -> Result<f64> {
        self.value_opt(nid, component).ok_or_else(|| {
            PyrucastError::Message(format!(
                "no subfield defines (node {}, component {})",
                nid, component
            ))
        })
    }

    /// Like [`NodeFieldView::value`], `None` when absent. Each zone's node
    /// lookup goes through its support's cached `NodeId → index` map (held
    /// open by `supports`), so the scan that used to dominate `deformation`
    /// is now O(1).
    pub(crate) fn value_opt(&self, nid: NodeId, component: &str) -> Option<f64> {
        self.inner
            .zones
            .iter()
            .zip(&self.supports)
            .find_map(|(zone, support)| {
                let ni = *support.node_index().get(&nid)?;
                zone.component_value_at(ni, component)
            })
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

// ─── Archive ────────────────────────────────────────────────────────────────

impl crate::archive::Archivable for SubNodeField {
    const TAG: &'static str = "SubNodeField";

    /// Take the node list from the support again rather than carrying a copy of
    /// it in the file. The support is a sealed POI1 mesh, so its connectivity
    /// *is* the node list, in the order this field indexes its values by.
    fn on_load(&mut self) {
        self.nodes = self.support.read().connectivity().to_vec();
    }
}

impl crate::archive::Archivable for NodeField {
    const TAG: &'static str = "NodeField";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Node;
    use crate::handle::Handle;

    fn make_poi1_with(n_nodes: usize) -> (Handle<Coords>, Vec<Node>, Handle<SubMesh>) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let nodes: Vec<Node> = (0..n_nodes)
            .map(|i| Node::create_in(coords.clone(), &[i as f64, 0.0]).unwrap())
            .collect();
        let sm_handle = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            for n in &nodes {
                sm.add_cell(&[n.id()]).unwrap();
            }
            Handle::new(sm)
        };
        (coords, nodes, sm_handle)
    }

    #[test]
    fn from_poi1_zero_initialized() {
        let (_cfg, _nodes, sm) = make_poi1_with(3);
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert_eq!(f.node_count(), 3);
        assert_eq!(f.component_count(), 1);
        for i in 0..3 {
            assert_eq!(f.get(i, 0).unwrap(), 0.0);
        }
    }

    #[test]
    fn from_poi1_rejects_non_poi1() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let sm = Handle::new(SubMesh::new(coords, ElementType::SEG2));
        let err = SubNodeField::from_poi1(&sm, vec!["X".into()]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn from_poi1_rejects_empty_components() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let err = SubNodeField::from_poi1(&sm, vec![]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn from_poi1_rejects_duplicate_components() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let err = SubNodeField::from_poi1(&sm, vec!["UX".into(), "UX".into()]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn shares_support_refcounts() {
        let (coords, nodes, sm) = make_poi1_with(2);
        // Each node has refcount = 2 (Node + SubMesh).
        {
            let c = coords.read();
            assert_eq!(c.refcount(nodes[0].id()), 2);
            assert_eq!(c.refcount(nodes[1].id()), 2);
        }
        let f = SubNodeField::from_poi1(&sm, vec!["P".into()]).unwrap();
        // The field shares the SubMesh handle, so per-node refcounts
        // are unchanged.
        {
            let c = coords.read();
            assert_eq!(c.refcount(nodes[0].id()), 2);
            assert_eq!(c.refcount(nodes[1].id()), 2);
        }
        drop(f);
        {
            let c = coords.read();
            assert_eq!(c.refcount(nodes[0].id()), 2);
            assert_eq!(c.refcount(nodes[1].id()), 2);
        }
    }

    #[test]
    fn get_set_multi_component() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let mut f =
            SubNodeField::from_poi1(&sm, vec!["UX".into(), "UY".into(), "UZ".into()]).unwrap();
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
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into(), "P".into()]).unwrap();
        let ci_p = f.component_index("P").unwrap();
        f.set_by_node(nodes[1].id(), ci_p, 42.0).unwrap();
        assert_eq!(f.get_by_node(nodes[1].id(), ci_p).unwrap(), 42.0);
    }

    #[test]
    fn out_of_bounds_errors() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["X".into()]).unwrap();
        assert!(f.get(5, 0).is_err());
        assert!(f.get(0, 5).is_err());
        assert!(f.set(5, 0, 1.0).is_err());
        assert!(f.node_values(5).is_err());
    }

    #[test]
    fn unknown_node_or_component_errors() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert!(f.get_by_node(NodeId(999), 0).is_err());
        assert_eq!(f.component_index("missing"), None);
    }

    #[test]
    fn protects_nodes_from_gc_after_node_and_submesh_drop() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let n = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let nid = n.id();
        let sm_handle = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[nid]).unwrap();
            Handle::new(sm)
        };
        let field = SubNodeField::from_poi1(&sm_handle, vec!["T".into()]).unwrap();
        // Drop the Node and the user's SubMesh handle; the field still
        // holds a clone of the SubMesh handle, which keeps the SubMesh
        // alive, which keeps the nodes alive.
        drop(n);
        drop(sm_handle);
        assert_eq!(coords.write().gc(), 0);
        assert!(coords.read().is_alive(nid));
        drop(field);
        assert_eq!(coords.write().gc(), 1);
        assert!(!coords.read().is_alive(nid));
    }

    #[test]
    fn support_submesh_mirrors_field_nodes() {
        let (coords, nodes, sm) = make_poi1_with(3);
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        // refcount before: Node + SubMesh = 2 each (the field shares
        // the user's SubMesh, no extra per-node incref).
        assert_eq!(coords.read().refcount(nodes[0].id()), 2);

        let sm2 = f.support_submesh().unwrap();
        // the freshly built SubMesh adds one incref each → 3
        assert_eq!(coords.read().refcount(nodes[0].id()), 3);

        assert_eq!(sm2.cell_count(), 3);
        let conn = sm2.connectivity();
        for (i, n) in nodes.iter().enumerate() {
            assert_eq!(conn[i], n.id());
        }

        drop(sm2);
        // back to 2
        assert_eq!(coords.read().refcount(nodes[0].id()), 2);
    }

    #[test]
    fn debug_and_display() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let f = SubNodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        let d = format!("{:?}", f);
        assert!(d.contains("SubNodeField"));
        assert!(d.contains("UX"));
        let s = format!("{}", f);
        assert!(s.contains("SubNodeField"));
        assert!(s.contains("2 node(s)"));
        assert!(s.contains("2 component(s)"));
        assert!(s.contains("UX, UY"));
    }

    // ── Index ────────────────────────────────────────────────────────────────

    #[test]
    fn index_read_write() {
        let (_cfg, nodes, sm) = make_poi1_with(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
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
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let _ = f[(NodeId(999), "T")];
    }

    #[test]
    #[should_panic]
    fn index_unknown_component_panics() {
        let (_cfg, nodes, sm) = make_poi1_with(1);
        let f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        let _ = f[(nodes[0].id(), "X")];
    }

    // ── Accès idiomatique ────────────────────────────────────────────────────

    #[test]
    fn value_and_set_value() {
        let (_cfg, nodes, sm) = make_poi1_with(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
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
        let mut f = SubNodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
        f.set_value(nodes[0].id(), "UX", 1.25).unwrap();

        let dumped = f.render(&DumpOptions::default());
        assert!(
            dumped.contains("UX") && dumped.contains("UY"),
            "headers:\n{dumped}"
        );
        assert!(
            dumped.contains("1.250"),
            "value at default precision:\n{dumped}"
        );
        assert_eq!(
            dumped.lines().count(),
            4,
            "summary + header + 2 rows:\n{dumped}"
        );

        // Debug must stay bounded: structure, never the value buffer.
        let dbg = format!("{f:?}");
        assert!(dbg.contains("node_count"), "{dbg}");
        assert!(!dbg.contains("1.25"), "Debug must not leak values: {dbg}");
    }

    // ── NodeField (agrégat) ─────────────────────────────────────────────────

    /// Two TRI3 zones sharing an interface edge (nodes n1, n2).
    fn make_two_zone_mesh() -> (Handle<Coords>, Vec<Node>, Mesh) {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
        let n3 = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm_a = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[n0.id(), n1.id(), n2.id()]).unwrap();
            Handle::new(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
            sm.add_cell(&[n1.id(), n3.id(), n2.id()]).unwrap();
            Handle::new(sm)
        };
        mesh.add_sub(sm_a).unwrap();
        mesh.add_sub(sm_b).unwrap();
        (coords, vec![n0, n1, n2, n3], mesh)
    }

    #[test]
    fn nf_new_one_sub_per_submesh() {
        let (_cfg, nodes, mesh) = make_two_zone_mesh();
        let f = NodeField::new(&mesh, vec!["T".into()]).unwrap();
        assert_eq!(f.len(), 2);
        // 4 distinct nodes; interface nodes n1, n2 stored in both subs.
        assert_eq!(f.node_count().unwrap(), 4);
        assert_eq!(f.get(0).unwrap().read().node_count(), 3);
        assert_eq!(f.get(1).unwrap().read().node_count(), 3);
        // Zero-initialized everywhere.
        assert_eq!(f.value(nodes[1].id(), "T").unwrap(), 0.0);
    }

    #[test]
    fn nf_with_per_zone_components() {
        let (_cfg, nodes, mesh) = make_two_zone_mesh();
        let f =
            NodeField::with(&mesh, &[vec!["T".into()], vec!["UX".into(), "UY".into()]]).unwrap();
        use crate::containers::field::Field;
        assert_eq!(Field::components(&f).unwrap(), vec!["T", "UX", "UY"]);
        // T exists on zone 0 only: defined at n0, absent at n3.
        assert_eq!(f.value(nodes[0].id(), "T").unwrap(), 0.0);
        assert!(f.value(nodes[3].id(), "T").is_err());
        assert_eq!(f.value_opt(nodes[3].id(), "T").unwrap(), None);
    }

    #[test]
    fn nf_with_rejects_mismatched_length() {
        let (_cfg, _nodes, mesh) = make_two_zone_mesh();
        assert!(NodeField::with(&mesh, &[vec!["T".into()]]).is_err());
    }

    #[test]
    fn nf_value_first_sub_wins() {
        let (_cfg, nodes, mesh) = make_two_zone_mesh();
        let f = NodeField::new(&mesh, vec!["T".into()]).unwrap();
        let interface = nodes[1].id();
        // Diverging interface values: reads pick sub 0, check() errors.
        f.get(0)
            .unwrap()
            .write()
            .set_value(interface, "T", 1.0)
            .unwrap();
        f.get(1)
            .unwrap()
            .write()
            .set_value(interface, "T", 2.0)
            .unwrap();
        assert_eq!(f.value(interface, "T").unwrap(), 1.0);
        assert!(f.check().is_err());
        // Re-aligned values: check() passes.
        f.get(1)
            .unwrap()
            .write()
            .set_value(interface, "T", 1.0)
            .unwrap();
        f.check().unwrap();
    }

    #[test]
    fn nf_values_at_preserves_order_and_errors_on_absent() {
        let (_cfg, nodes, mesh) = make_two_zone_mesh();
        let f = NodeField::new(&mesh, vec!["T".into()]).unwrap();
        {
            // n0 and n2 both live in zone 0 (the TRI3 n0-n1-n2).
            let mut z0 = f.get(0).unwrap().write();
            z0.set_value(nodes[0].id(), "T", 10.0).unwrap();
            z0.set_value(nodes[2].id(), "T", 30.0).unwrap();
        }
        // Same order as asked, duplicates kept.
        let ids = [nodes[2].id(), nodes[0].id(), nodes[2].id()];
        assert_eq!(f.values_at(&ids, "T").unwrap(), vec![30.0, 10.0, 30.0]);
        // Empty query → empty result.
        assert_eq!(f.values_at(&[], "T").unwrap(), Vec::<f64>::new());
        // Unknown component → error (mirrors `value`).
        assert!(f.values_at(&[nodes[0].id()], "P").is_err());
    }

    #[test]
    fn nf_check_ok_on_disjoint_components() {
        let (_cfg, _nodes, mesh) = make_two_zone_mesh();
        // Same interface nodes, but disjoint components: no duplicate pair.
        let f = NodeField::with(&mesh, &[vec!["T".into()], vec!["P".into()]]).unwrap();
        f.check().unwrap();
    }

    #[test]
    fn nf_add_is_structural_merge() {
        let (_cfg, _nodes, mesh) = make_two_zone_mesh();
        let a = NodeField::from_submesh(&mesh.get(0).unwrap(), vec!["T".into()]).unwrap();
        let b = NodeField::from_submesh(&mesh.get(1).unwrap(), vec!["P".into()]).unwrap();
        let c = a.union(&b).unwrap();
        assert_eq!(c.len(), 2);
        // Sub-handles are shared, not copied.
        assert!(c.get(0).unwrap().same_object(&a.get(0).unwrap()));
    }

    #[test]
    fn nf_check_push_rejects_mismatched_cfg() {
        let (_cfg, _nodes, mesh) = make_two_zone_mesh();
        let mut f = NodeField::new(&mesh, vec!["T".into()]).unwrap();
        let cfg2 = Handle::new(Coords::new(1).unwrap());
        let n = Node::create_in(cfg2.clone(), &[0.0]).unwrap();
        let sm = {
            let mut sm = SubMesh::new(cfg2, ElementType::POI1);
            sm.add_cell(&[n.id()]).unwrap();
            Handle::new(sm)
        };
        let alien = Handle::new(SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap());
        assert!(f.add_sub(alien).is_err());
    }

    #[test]
    fn nf_from_support_poi1_shares_handle() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let f = NodeField::from_submesh(&sm, vec!["T".into()]).unwrap();
        let support = f.get(0).unwrap().read().support();
        assert!(support.same_object(&sm));
    }

    #[test]
    fn nf_new_rejects_empty_mesh_or_components() {
        let (_cfg, _nodes, mesh) = make_two_zone_mesh();
        assert!(NodeField::new(&Mesh::empty(), vec!["T".into()]).is_err());
        assert!(NodeField::new(&mesh, vec![]).is_err());
        assert!(NodeField::new(&mesh, vec!["T".into(), "T".into()]).is_err());
    }

    // ── Scalaires sur composante ──────────────────────────────────────────────

    #[test]
    fn component_scalar_ops() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
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
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert!(f.div_to_component("T", 0.0).is_err());
    }

    #[test]
    fn component_scalar_unknown_component_errors() {
        let (_cfg, _nodes, sm) = make_poi1_with(1);
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        assert!(f.add_to_component("X", 1.0).is_err());
        assert!(f.sub_to_component("X", 1.0).is_err());
        assert!(f.mul_to_component("X", 1.0).is_err());
        assert!(f.div_to_component("X", 1.0).is_err());
    }

    // ── Clone ────────────────────────────────────────────────────────────────

    #[test]
    fn clone_is_independent() {
        let (coords, nodes, sm) = make_poi1_with(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set(0, 0, 42.0).unwrap();
        let g = f.clone();
        // Both fields share the same SubMesh handle, so per-node
        // refcounts in the Coords stay at 2 (Node + SubMesh).
        assert_eq!(coords.read().refcount(nodes[0].id()), 2);
        // Mutation of f does not affect g (values are independent).
        f.set(0, 0, 99.0).unwrap();
        assert_eq!(g.get(0, 0).unwrap(), 42.0);
        drop(g);
        assert_eq!(coords.read().refcount(nodes[0].id()), 2);
    }

    // ── Opérateurs +,-,*,/ avec f64 ─────────────────────────────────────────

    #[test]
    fn operator_add_f64() {
        let (_cfg, _nodes, sm) = make_poi1_with(2);
        let mut f = SubNodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
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
        let mut f = SubNodeField::from_poi1(&sm, vec!["T".into()]).unwrap();
        f.set(0, 0, 12.0).unwrap();
        assert_eq!((f.clone() - 2.0).get(0, 0).unwrap(), 10.0);
        assert_eq!((f.clone() * 3.0).get(0, 0).unwrap(), 36.0);
        assert_eq!((f * 4.0 / 2.0).get(0, 0).unwrap(), 24.0);
    }
}
