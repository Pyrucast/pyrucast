//! Node — RAII accessor to a node of a [`Configuration`].
//!
//! `Node` is the **user-facing interface** to a node: it holds a handle to
//! its `Configuration` and a node id, and automatically maintains the
//! **internal** node refcount (`Clone` increments, `Drop` decrements). As
//! long as at least one `Node` exists, the node is protected from the
//! `Configuration`'s garbage collector.
//!
//! Internal code can still manipulate [`crate::containers::mesh::NodeId`]
//! values directly, but then loses the automatic GC protection: it must
//! call [`Configuration::incref`](crate::containers::mesh::Configuration::incref) /
//! [`Configuration::decref`](crate::containers::mesh::Configuration::decref)
//! by hand.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::Configuration;
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::{insert, with, with_mut};
//!
//! let cfg = insert(Configuration::new(2).unwrap());
//! let n = Node::create_in(cfg.clone(), &[1.0, 2.0]).unwrap();
//! assert_eq!(n.coord().unwrap(), vec![1.0, 2.0]);
//!
//! // The GC does not touch a node that a live Node still references.
//! with_mut(&cfg, |c| assert_eq!(c.gc(), 0)).unwrap();
//!
//! let id = n.id();
//! drop(n);
//! // Now the refcount is 0; gc collects.
//! with_mut(&cfg, |c| assert_eq!(c.gc(), 1)).unwrap();
//! with(&cfg, |c| assert!(!c.is_alive(id))).unwrap();
//! ```

use crate::containers::mesh::{Configuration, NodeId};
use crate::error::Result;
use crate::store::{with, with_mut, Handle};
use std::fmt;

/// RAII accessor to a node of a `Configuration`.
pub struct Node {
    handle: Handle<Configuration>,
    id: NodeId,
}

impl Node {
    /// Add a new node to the pointed `Configuration` and return a `Node`
    /// referencing it (refcount = 1).
    pub fn create_in(cfg: Handle<Configuration>, coords: &[f64]) -> Result<Self> {
        // `add_node` initializes refcount = 1; this Node takes that unit.
        let id = with_mut(&cfg, |c| c.add_node(coords))??;
        Ok(Self { handle: cfg, id })
    }

    /// Build an additional `Node` for an existing id (refcount += 1).
    pub fn acquire(cfg: Handle<Configuration>, id: NodeId) -> Result<Self> {
        with_mut(&cfg, |c| c.incref(id))??;
        Ok(Self { handle: cfg, id })
    }

    /// Build a `Node` from a handle and an id that have **already been
    /// incremented** on the Configuration side. Internal escape hatch used
    /// by FFI wrappers that have already paid the refcount increment.
    #[cfg(feature = "python-api")]
    pub(crate) fn from_parts(handle: Handle<Configuration>, id: NodeId) -> Self {
        Self { handle, id }
    }

    /// Internal identifier of the node.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Handle to the owning `Configuration` (internal clone).
    pub fn configuration(&self) -> Handle<Configuration> {
        self.handle.clone()
    }

    /// Coordinates (copied) in the `Configuration`'s active set.
    pub fn coord(&self) -> Result<Vec<f64>> {
        let v = with(&self.handle, |c| c.coord(self.id).map(|s| s.to_vec()))??;
        Ok(v)
    }

    /// Set the coordinates of the node in the active set.
    pub fn set_coord(&self, coords: &[f64]) -> Result<()> {
        with_mut(&self.handle, |c| c.set_coord(self.id, coords))??;
        Ok(())
    }
}

impl Clone for Node {
    fn clone(&self) -> Self {
        let _ = with_mut(&self.handle, |c| c.incref(self.id));
        Self {
            handle: self.handle.clone(),
            id: self.id,
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        let _ = with_mut(&self.handle, |c| c.decref(self.id));
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Node");
        s.field("id", &self.id).field("handle", &self.handle);
        // Debug may take the lock to surface the actual values (Display
        // stays lock-free). If the node is no longer reachable, fall back
        // to the error rather than failing to format.
        match self.coord() {
            Ok(coord) => s.field("coord", &coord),
            Err(e) => s.field("coord", &format_args!("<{e}>")),
        };
        s.finish()
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display stays lock-free; use Debug or `coord()` for the values.
        write!(f, "<Node #{}>", self.id)
    }
}

impl crate::dump::Dump for Node {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        match self.coord() {
            Ok(c) => {
                let coords: Vec<String> = c
                    .iter()
                    .map(|v| crate::dump::fmt_float(*v, opts.precision))
                    .collect();
                format!("Node #{} @ ({})", self.id, coords.join(", "))
            }
            Err(e) => format!("Node #{} <{e}>", self.id),
        }
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::insert;

    #[test]
    fn node_protects_from_gc() {
        let cfg = insert(Configuration::new(2).unwrap());
        let n = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let id = n.id();
        with_mut(&cfg, |c| assert_eq!(c.gc(), 0)).unwrap();
        with(&cfg, |c| assert!(c.is_alive(id))).unwrap();
        drop(n);
        with_mut(&cfg, |c| assert_eq!(c.gc(), 1)).unwrap();
        with(&cfg, |c| assert!(!c.is_alive(id))).unwrap();
    }

    #[test]
    fn clone_and_drop_maintain_refcount() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n = Node::create_in(cfg.clone(), &[1.0]).unwrap();
        let id = n.id();
        let m = n.clone();
        with(&cfg, |c| assert_eq!(c.refcount(id), 2)).unwrap();
        drop(n);
        with(&cfg, |c| assert_eq!(c.refcount(id), 1)).unwrap();
        drop(m);
        with(&cfg, |c| assert_eq!(c.refcount(id), 0)).unwrap();
    }

    #[test]
    fn acquire_shares_same_id() {
        let cfg = insert(Configuration::new(1).unwrap());
        let n = Node::create_in(cfg.clone(), &[7.0]).unwrap();
        let id = n.id();
        let m = Node::acquire(cfg.clone(), id).unwrap();
        assert_eq!(n.id(), m.id());
        with(&cfg, |c| assert_eq!(c.refcount(id), 2)).unwrap();
        drop(n);
        drop(m);
        with(&cfg, |c| assert_eq!(c.refcount(id), 0)).unwrap();
    }

    #[test]
    fn coord_and_set_coord() {
        let cfg = insert(Configuration::new(2).unwrap());
        let n = Node::create_in(cfg, &[1.0, 2.0]).unwrap();
        assert_eq!(n.coord().unwrap(), vec![1.0, 2.0]);
        n.set_coord(&[5.0, 6.0]).unwrap();
        assert_eq!(n.coord().unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn debug_and_display() {
        let cfg = insert(Configuration::new(2).unwrap());
        let n = Node::create_in(cfg, &[1.5, 2.5]).unwrap();
        let d = format!("{:?}", n);
        assert!(d.contains("Node"));
        assert!(d.contains("coord"));
        assert!(d.contains("1.5") && d.contains("2.5"));
        let s = format!("{}", n);
        assert!(s.starts_with("<Node #"));
    }
}
