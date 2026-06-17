//! Node — RAII accessor to a node of a [`Coords`].
//!
//! `Node` is the **user-facing interface** to a node: it holds a handle to
//! its `Coords` and a node id, and automatically maintains the
//! **internal** node refcount (`Clone` increments, `Drop` decrements). As
//! long as at least one `Node` exists, the node is protected from the
//! `Coords`'s garbage collector.
//!
//! Internal code can still manipulate [`crate::containers::mesh::NodeId`]
//! values directly, but then loses the automatic GC protection: it must
//! call [`Coords::incref`](crate::containers::mesh::Coords::incref) /
//! [`Coords::decref`](crate::containers::mesh::Coords::decref)
//! by hand.
//!
//! # Example
//!
//! ```
//! use pyrucast::containers::mesh::Coords;
//! use pyrucast::containers::mesh::Node;
//! use pyrucast::store::{insert, read, write};
//!
//! let coords = insert(Coords::new(2).unwrap());
//! let n = Node::create_in(coords.clone(), &[1.0, 2.0]).unwrap();
//! assert_eq!(n.coord().unwrap(), vec![1.0, 2.0]);
//!
//! // The GC does not touch a node that a live Node still references.
//! assert_eq!(write(&coords).unwrap().gc(), 0);
//!
//! let id = n.id();
//! drop(n);
//! // Now the refcount is 0; gc collects.
//! assert_eq!(write(&coords).unwrap().gc(), 1);
//! assert!(!read(&coords).unwrap().is_alive(id));
//! ```

use crate::containers::mesh::{Coords, NodeId};
use crate::error::Result;
use crate::store::{read, write, Handle};
use std::fmt;

/// RAII accessor to a node of a `Coords`.
pub struct Node {
    handle: Handle<Coords>,
    id: NodeId,
}

impl Node {
    /// Add a new node to the pointed `Coords` and return a `Node`
    /// referencing it (refcount = 1).
    pub fn create_in(coords: Handle<Coords>, coord: &[f64]) -> Result<Self> {
        // `add_node` initializes refcount = 1; this Node takes that unit.
        let id = write(&coords)?.add_node(coord)?;
        Ok(Self { handle: coords, id })
    }

    /// Build an additional `Node` for an existing id (refcount += 1).
    pub fn acquire(coords: Handle<Coords>, id: NodeId) -> Result<Self> {
        write(&coords)?.incref(id)?;
        Ok(Self { handle: coords, id })
    }

    /// Build a `Node` from a handle and an id that have **already been
    /// incremented** on the Coords side. Internal escape hatch used
    /// by FFI wrappers that have already paid the refcount increment.
    #[cfg(feature = "python-api")]
    pub(crate) fn from_parts(handle: Handle<Coords>, id: NodeId) -> Self {
        Self { handle, id }
    }

    /// Internal identifier of the node.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Handle to the owning `Coords` (internal clone).
    pub fn coords(&self) -> Handle<Coords> {
        self.handle.clone()
    }

    /// Coordinates (copied) in the `Coords`'s active set.
    pub fn coord(&self) -> Result<Vec<f64>> {
        Ok(read(&self.handle)?.coord(self.id)?.to_vec())
    }

    /// Set the coordinates of the node in the active set.
    pub fn set_coord(&self, coords: &[f64]) -> Result<()> {
        write(&self.handle)?.set_coord(self.id, coords)?;
        Ok(())
    }
}

impl Clone for Node {
    fn clone(&self) -> Self {
        if let Ok(mut c) = write(&self.handle) {
            let _ = c.incref(self.id);
        }
        Self {
            handle: self.handle.clone(),
            id: self.id,
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        if let Ok(mut c) = write(&self.handle) {
            let _ = c.decref(self.id);
        }
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
        let coords = insert(Coords::new(2).unwrap());
        let n = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let id = n.id();
        assert_eq!(write(&coords).unwrap().gc(), 0);
        assert!(read(&coords).unwrap().is_alive(id));
        drop(n);
        assert_eq!(write(&coords).unwrap().gc(), 1);
        assert!(!read(&coords).unwrap().is_alive(id));
    }

    #[test]
    fn clone_and_drop_maintain_refcount() {
        let coords = insert(Coords::new(1).unwrap());
        let n = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let id = n.id();
        let m = n.clone();
        assert_eq!(read(&coords).unwrap().refcount(id), 2);
        drop(n);
        assert_eq!(read(&coords).unwrap().refcount(id), 1);
        drop(m);
        assert_eq!(read(&coords).unwrap().refcount(id), 0);
    }

    #[test]
    fn acquire_shares_same_id() {
        let coords = insert(Coords::new(1).unwrap());
        let n = Node::create_in(coords.clone(), &[7.0]).unwrap();
        let id = n.id();
        let m = Node::acquire(coords.clone(), id).unwrap();
        assert_eq!(n.id(), m.id());
        assert_eq!(read(&coords).unwrap().refcount(id), 2);
        drop(n);
        drop(m);
        assert_eq!(read(&coords).unwrap().refcount(id), 0);
    }

    #[test]
    fn coord_and_set_coord() {
        let coords = insert(Coords::new(2).unwrap());
        let n = Node::create_in(coords, &[1.0, 2.0]).unwrap();
        assert_eq!(n.coord().unwrap(), vec![1.0, 2.0]);
        n.set_coord(&[5.0, 6.0]).unwrap();
        assert_eq!(n.coord().unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn debug_and_display() {
        let coords = insert(Coords::new(2).unwrap());
        let n = Node::create_in(coords, &[1.5, 2.5]).unwrap();
        let d = format!("{:?}", n);
        assert!(d.contains("Node"));
        assert!(d.contains("coord"));
        assert!(d.contains("1.5") && d.contains("2.5"));
        let s = format!("{}", n);
        assert!(s.starts_with("<Node #"));
    }
}
