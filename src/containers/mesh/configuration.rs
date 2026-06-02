//! Configuration — sets of node coordinates with garbage collection.
//!
//! A [`Configuration`] holds **one or more sets of coordinates** for the
//! same set of nodes, in a fixed dimension.
//!
//! # Node identity
//!
//! Every created node receives a **stable** internal identifier
//! ([`NodeId`]), unique for the lifetime of the `Configuration`: **no id
//! is ever reused**, even after garbage collection. Other objects (meshes,
//! fields) can therefore reference a node by id without worrying about
//! stability.
//!
//! # Deletion policy: no direct removal
//!
//! There is **no** `remove_node` method. A referenced node is protected.
//! Only the garbage collector [`Configuration::gc`] reclaims nodes whose
//! **internal** refcount has reached 0.
//!
//! # Two-level refcount model
//!
//! - The **Configuration slot** in the global store is protected by the
//!   usual [`crate::store::Handle`] refcount.
//! - **Each node** inside the Configuration has its own refcount,
//!   manipulated via [`Configuration::incref`] / [`Configuration::decref`]
//!   (used by [`crate::containers::mesh::Node`] and, later, by meshes and fields).
//!
//! # Identity vs solver ordering
//!
//! An optional permutation (`Vec<u32>`) separates the **solver order**
//! from the **identity**: `permutation[node_id]` is the solver-order index
//! assigned to `node_id`. Phase 4 (Cuthill–McKee renumbering) will
//! recompute it; the identity (`NodeId`) is never modified.
//!
//! # Multiple coordinate sets
//!
//! Useful for switching between reference / deformed / predicted
//! configurations. An active set is designated by index;
//! [`Configuration::coord`] reads from the active set.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::{Configuration, NodeId};
//! use pyrucast::store::{insert, with, with_mut};
//!
//! let h = insert(Configuration::new(2).unwrap());
//! let a: NodeId = with_mut(&h, |c| c.add_node(&[0.0, 0.0])).unwrap().unwrap();
//! // add_node initializes refcount = 1: without decref, the node is protected.
//! with_mut(&h, |c| { assert_eq!(c.gc(), 0); }).unwrap();
//! // After decref, refcount drops to 0 and gc collects it.
//! with_mut(&h, |c| { c.decref(a).unwrap(); assert_eq!(c.gc(), 1); }).unwrap();
//! ```

use crate::error::{PyrucastError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Stable internal identifier of a node inside a `Configuration`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl crate::dump::Dump for NodeId {
    fn render(&self, _opts: &crate::dump::DumpOptions) -> String {
        self.to_string()
    }
}

/// Sets of node coordinates with stable identity, multi-set support,
/// optional solver permutation, and a garbage collector for unreferenced
/// nodes.
#[derive(Serialize, Deserialize)]
pub struct Configuration {
    dim: u8,
    /// `coord_sets[s][id * dim + k]` — each set holds `capacity * dim` values.
    coord_sets: Vec<Vec<f64>>,
    set_names: Vec<String>,
    active: usize,
    /// `alive[id] == false` ⇒ collected by the GC. Once `false`, stays so forever.
    alive: Vec<bool>,
    /// Per-node refcount. The GC collects `alive` nodes whose refcount is 0.
    refcount: Vec<u32>,
    /// Solver permutation (length == capacity) or `None` for identity.
    permutation: Option<Vec<u32>>,
}

impl Configuration {
    /// Create an empty configuration in dimension `dim` (≥ 1). A first set
    /// named `"default"` is created automatically.
    pub fn new(dim: u8) -> Result<Self> {
        if dim == 0 {
            return Err(PyrucastError::Message("dim must be ≥ 1".into()));
        }
        Ok(Self {
            dim,
            coord_sets: vec![Vec::new()],
            set_names: vec!["default".into()],
            active: 0,
            alive: Vec::new(),
            refcount: Vec::new(),
            permutation: None,
        })
    }

    /// Geometric dimension.
    pub fn dim(&self) -> u8 {
        self.dim
    }

    /// Number of live (not collected) nodes.
    pub fn node_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    /// Capacity (total slots, live + collected). Never decreases.
    pub fn capacity(&self) -> usize {
        self.alive.len()
    }

    /// Whether a node is still alive.
    pub fn is_alive(&self, id: NodeId) -> bool {
        self.alive.get(id.0 as usize).copied().unwrap_or(false)
    }

    /// Add a node with these coordinates in **all** sets. Initializes its
    /// refcount to 1 — the caller is responsible for at least one decrement
    /// (typically through the end-of-life of a [`crate::containers::mesh::Node`]).
    pub fn add_node(&mut self, coords: &[f64]) -> Result<NodeId> {
        if coords.len() != self.dim as usize {
            return Err(PyrucastError::Message(format!(
                "add_node: expected {} coordinates, got {}",
                self.dim,
                coords.len()
            )));
        }
        let id = self.alive.len() as u32;
        for set in &mut self.coord_sets {
            set.extend_from_slice(coords);
        }
        self.alive.push(true);
        self.refcount.push(1);
        if let Some(perm) = &mut self.permutation {
            perm.push(id);
        }
        Ok(NodeId(id))
    }

    /// Increment the refcount of a live node.
    pub fn incref(&mut self, id: NodeId) -> Result<()> {
        self.ensure_alive(id)?;
        let r = &mut self.refcount[id.0 as usize];
        *r = r.saturating_add(1);
        Ok(())
    }

    /// Decrement the refcount of a live node. The node is not immediately
    /// collected even if the refcount reaches 0: call
    /// [`Configuration::gc`] for that.
    pub fn decref(&mut self, id: NodeId) -> Result<()> {
        self.ensure_alive(id)?;
        let r = &mut self.refcount[id.0 as usize];
        if *r == 0 {
            return Err(PyrucastError::Message(format!(
                "decref: refcount already zero for node {}",
                id.0
            )));
        }
        *r -= 1;
        Ok(())
    }

    /// Current refcount of a node (0 for a collected or unknown node).
    pub fn refcount(&self, id: NodeId) -> u32 {
        if self.is_alive(id) {
            self.refcount[id.0 as usize]
        } else {
            0
        }
    }

    /// Garbage collector: mark as collected every live node whose refcount
    /// is 0. Returns the number of collected nodes. Ids are never reused.
    pub fn gc(&mut self) -> usize {
        let mut collected = 0;
        for i in 0..self.alive.len() {
            if self.alive[i] && self.refcount[i] == 0 {
                self.alive[i] = false;
                collected += 1;
            }
        }
        collected
    }

    /// Coordinates of a node in the active set. Error if the node was
    /// collected or never existed.
    pub fn coord(&self, id: NodeId) -> Result<&[f64]> {
        self.ensure_alive(id)?;
        let d = self.dim as usize;
        let s = id.0 as usize * d;
        Ok(&self.coord_sets[self.active][s..s + d])
    }

    /// Set the coordinates of a node in the active set.
    pub fn set_coord(&mut self, id: NodeId, coords: &[f64]) -> Result<()> {
        self.ensure_alive(id)?;
        if coords.len() != self.dim as usize {
            return Err(PyrucastError::Message(format!(
                "set_coord: expected {} coordinates, got {}",
                self.dim,
                coords.len()
            )));
        }
        let d = self.dim as usize;
        let s = id.0 as usize * d;
        self.coord_sets[self.active][s..s + d].copy_from_slice(coords);
        Ok(())
    }

    fn ensure_alive(&self, id: NodeId) -> Result<()> {
        if !self.is_alive(id) {
            return Err(PyrucastError::Message(format!(
                "node {} not found or collected",
                id.0
            )));
        }
        Ok(())
    }

    /// Iterate over live NodeIds in ascending id order.
    pub fn iter_live(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(NodeId(i as u32)))
    }

    // ─── Coordinate sets ───

    /// Add a new coordinate set by cloning the active one. Returns its index.
    pub fn add_coord_set(&mut self, name: impl Into<String>) -> usize {
        let copy = self.coord_sets[self.active].clone();
        self.coord_sets.push(copy);
        self.set_names.push(name.into());
        self.coord_sets.len() - 1
    }

    /// Switch the active set to index `set`.
    pub fn switch_to(&mut self, set: usize) -> Result<()> {
        if set >= self.coord_sets.len() {
            return Err(PyrucastError::Message(format!(
                "switch_to: index {} ≥ set count ({})",
                set,
                self.coord_sets.len()
            )));
        }
        self.active = set;
        Ok(())
    }

    /// Index of the active set.
    pub fn active_set(&self) -> usize {
        self.active
    }

    /// Names of the sets, in order.
    pub fn set_names(&self) -> &[String] {
        &self.set_names
    }

    // ─── Solver permutation ───

    /// Current permutation (length = capacity), or `None` for identity.
    pub fn permutation(&self) -> Option<&[u32]> {
        self.permutation.as_deref()
    }

    /// Set the solver permutation. Its length must equal `capacity`; each
    /// value must be unique and within `[0, capacity)`.
    pub fn set_permutation(&mut self, perm: Vec<u32>) -> Result<()> {
        let cap = self.capacity();
        if perm.len() != cap {
            return Err(PyrucastError::Message(format!(
                "set_permutation: length {} ≠ capacity {}",
                perm.len(),
                cap
            )));
        }
        let cap_u = cap as u32;
        let mut seen = vec![false; cap];
        for &v in &perm {
            if v >= cap_u {
                return Err(PyrucastError::Message(format!(
                    "set_permutation: value {} ≥ capacity {}",
                    v, cap_u
                )));
            }
            let i = v as usize;
            if seen[i] {
                return Err(PyrucastError::Message(format!(
                    "set_permutation: duplicate value {}",
                    v
                )));
            }
            seen[i] = true;
        }
        self.permutation = Some(perm);
        Ok(())
    }

    /// Clear the permutation (back to identity).
    pub fn clear_permutation(&mut self) {
        self.permutation = None;
    }
}

impl fmt::Debug for Configuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Configuration")
            .field("dim", &self.dim)
            .field("coord_sets", &self.set_names)
            .field("active", &self.active)
            .field("node_count", &self.node_count())
            .field("capacity", &self.capacity())
            .field("permutation", &self.permutation.is_some())
            .finish()
    }
}

impl fmt::Display for Configuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let active_name = &self.set_names[self.active];
        let collected = self.capacity() - self.node_count();
        let perm_label = if self.permutation.is_some() {
            "custom"
        } else {
            "identity"
        };
        write!(
            f,
            "Configuration: dim={}, sets={} (active=\"{}\"), nodes={} ({} collected), permutation: {}",
            self.dim,
            self.coord_sets.len(),
            active_name,
            self.node_count(),
            collected,
            perm_label
        )
    }
}

impl crate::dump::Dump for Configuration {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        use crate::dump::{fmt_float, table};
        let dim = self.dim as usize;
        const AXES: [&str; 3] = ["x", "y", "z"];
        let mut headers = vec!["node".to_string()];
        headers.extend((0..dim).map(|i| AXES.get(i).copied().unwrap_or("?").to_string()));
        headers.push("refs".to_string());
        let rows: Vec<Vec<String>> = self
            .iter_live()
            .map(|id| {
                let mut row = vec![id.to_string()];
                match self.coord(id) {
                    Ok(c) => row.extend(c.iter().map(|v| fmt_float(*v, opts.precision))),
                    Err(_) => row.extend((0..dim).map(|_| "?".to_string())),
                }
                row.push(self.refcount(id).to_string());
                row
            })
            .collect();
        format!("{self}\n{}", table(&headers, &rows, opts))
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_dim() {
        let c = Configuration::new(3).unwrap();
        assert_eq!(c.dim(), 3);
        assert_eq!(c.node_count(), 0);
        assert_eq!(c.capacity(), 0);
    }

    #[test]
    fn dim_zero_rejected() {
        let err = Configuration::new(0).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn add_node_initializes_refcount_to_one() {
        let mut c = Configuration::new(2).unwrap();
        let a = c.add_node(&[0.0, 0.0]).unwrap();
        assert_eq!(c.refcount(a), 1);
        assert!(c.is_alive(a));
    }

    #[test]
    fn add_node_invalid_dim() {
        let mut c = Configuration::new(3).unwrap();
        let err = c.add_node(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn gc_does_not_collect_referenced_nodes() {
        let mut c = Configuration::new(2).unwrap();
        let a = c.add_node(&[0.0, 0.0]).unwrap();
        // refcount = 1, gc must not collect anything
        assert_eq!(c.gc(), 0);
        assert!(c.is_alive(a));
        // After decref, refcount drops to 0
        c.decref(a).unwrap();
        assert_eq!(c.refcount(a), 0);
        // but only gc actually removes
        assert_eq!(c.gc(), 1);
        assert!(!c.is_alive(a));
    }

    #[test]
    fn incref_protects_from_gc() {
        let mut c = Configuration::new(1).unwrap();
        let a = c.add_node(&[3.0]).unwrap();
        c.incref(a).unwrap(); // refcount = 2
        c.decref(a).unwrap(); // refcount = 1
        assert_eq!(c.gc(), 0);
        c.decref(a).unwrap(); // refcount = 0
        assert_eq!(c.gc(), 1);
    }

    #[test]
    fn id_not_reused_after_gc() {
        let mut c = Configuration::new(1).unwrap();
        let a = c.add_node(&[0.0]).unwrap();
        c.decref(a).unwrap();
        c.gc();
        let b = c.add_node(&[1.0]).unwrap();
        assert_ne!(a.0, b.0);
        assert_eq!(b.0, 1);
        assert_eq!(c.capacity(), 2);
    }

    #[test]
    fn coord_after_gc_is_error() {
        let mut c = Configuration::new(1).unwrap();
        let a = c.add_node(&[42.0]).unwrap();
        c.decref(a).unwrap();
        c.gc();
        assert!(c.coord(a).is_err());
        assert!(c.set_coord(a, &[0.0]).is_err());
        assert!(c.incref(a).is_err());
        assert!(c.decref(a).is_err());
    }

    #[test]
    fn decref_at_zero_is_error() {
        let mut c = Configuration::new(1).unwrap();
        let a = c.add_node(&[0.0]).unwrap();
        c.decref(a).unwrap();
        let err = c.decref(a).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn set_coord_modifies_active_set() {
        let mut c = Configuration::new(2).unwrap();
        let a = c.add_node(&[0.0, 0.0]).unwrap();
        c.set_coord(a, &[3.0, 4.0]).unwrap();
        assert_eq!(c.coord(a).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn multiple_sets_and_switching() {
        let mut c = Configuration::new(2).unwrap();
        let a = c.add_node(&[0.0, 0.0]).unwrap();
        let s2 = c.add_coord_set("deformed");
        assert_eq!(s2, 1);
        c.switch_to(s2).unwrap();
        c.set_coord(a, &[10.0, 20.0]).unwrap();
        c.switch_to(0).unwrap();
        assert_eq!(c.coord(a).unwrap(), &[0.0, 0.0]);
        c.switch_to(1).unwrap();
        assert_eq!(c.coord(a).unwrap(), &[10.0, 20.0]);
        assert_eq!(c.set_names(), &["default".to_string(), "deformed".to_string()]);
    }

    #[test]
    fn switch_invalid() {
        let mut c = Configuration::new(2).unwrap();
        assert!(c.switch_to(5).is_err());
    }

    #[test]
    fn iter_live_skips_collected() {
        let mut c = Configuration::new(1).unwrap();
        let a = c.add_node(&[0.0]).unwrap();
        let b = c.add_node(&[1.0]).unwrap();
        let _cc = c.add_node(&[2.0]).unwrap();
        c.decref(b).unwrap();
        c.gc();
        let live: Vec<u32> = c.iter_live().map(|n| n.0).collect();
        assert_eq!(live, vec![a.0, 2]);
    }

    #[test]
    fn permutation_validation_and_invariant() {
        let mut c = Configuration::new(1).unwrap();
        for k in 0..4 {
            c.add_node(&[k as f64]).unwrap();
        }
        c.set_permutation(vec![3, 2, 1, 0]).unwrap();
        assert_eq!(c.permutation(), Some(&[3u32, 2, 1, 0][..]));
        assert!(c.set_permutation(vec![0, 0, 1, 2]).is_err());
        assert!(c.set_permutation(vec![0, 1, 2, 99]).is_err());
        assert!(c.set_permutation(vec![0, 1, 2]).is_err());
        c.clear_permutation();
        assert!(c.permutation().is_none());
    }

    #[test]
    fn permutation_extended_by_add_node() {
        let mut c = Configuration::new(1).unwrap();
        for k in 0..3 {
            c.add_node(&[k as f64]).unwrap();
        }
        c.set_permutation(vec![2, 1, 0]).unwrap();
        c.add_node(&[42.0]).unwrap();
        let perm = c.permutation().unwrap();
        assert_eq!(perm.len(), 4);
        let mut sorted = perm.to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn debug_display() {
        let mut c = Configuration::new(2).unwrap();
        c.add_node(&[0.0, 0.0]).unwrap();
        c.add_node(&[1.0, 1.0]).unwrap();
        let d = format!("{:?}", c);
        assert!(d.contains("Configuration"));
        assert!(d.contains("dim"));
        let s = format!("{}", c);
        assert!(s.contains("dim=2"));
        assert!(s.contains("nodes=2"));
        assert!(s.contains("identity"));
    }

    #[test]
    fn nodeid_display_and_debug() {
        let n = NodeId(7);
        assert_eq!(format!("{}", n), "7");
        assert_eq!(format!("{:?}", n), "NodeId(7)");
    }
}
