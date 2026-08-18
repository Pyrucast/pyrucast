//! Object handles — pyrucast's "object pile", cast3m-style.
//!
//! Every object the user manipulates lives behind a [`Handle<T>`]: a counted,
//! cheaply cloned reference carrying its own lock. There is no session to pass
//! around and no global registry — a handle *is* the object's address, and the
//! object dies with the last handle that names it.
//!
//! ```text
//! Handle<T>  =  Arc<RwLock<T>>  +  the API we want on it
//! ```
//!
//! # Guarantees
//!
//! - **Counted.** `Clone` shares, `Drop` releases; when the last handle goes,
//!   the value is dropped — running its own `Drop`, so side effects such as
//!   `SubMesh` releasing its nodes inside the `Coords` happen exactly once.
//! - **Always valid.** A handle cannot outlive its object: holding one keeps
//!   the object alive. There is no stale handle, no generation to check, and
//!   [`Handle::read`] / [`Handle::write`] cannot fail.
//! - **Identity is the pointer.** [`Handle::same_object`] answers « are these
//!   two references the same object? » — the basis of the aggregates' union.
//!
//! # Why not a slab of numbered slots
//!
//! There used to be one: a registry per [`std::any::TypeId`], a slab of
//! generational slots, a hand-written refcount, and a disk-swap path. Every
//! piece of it duplicated something `Arc` already does — the hand refcount
//! tracked what the `Arc`'s tracked, and the generation only protected handles
//! rebuilt from bytes, which the swap alone produced.
//!
//! What the registry offered beyond that was **enumeration** (« list every live
//! mesh ») and a **stable number** per object. Neither had ever reached a user.
//! Should enumeration become wanted — a cast3m-style listing, a whole-session
//! save — it costs a `Vec<Weak<_>>` registered in [`Handle::new`], which is why
//! that constructor is the single funnel through which objects are created.
//!
//! # Concurrency
//!
//! Locking is **per object**: each handle carries its own
//! [`parking_lot::RwLock`]. [`Handle::read`] returns a shared guard (many
//! concurrent readers), [`Handle::write`] an exclusive one. Guards are *owned*
//! (`'static`): they can be returned from functions and stored in structs, so
//! operators read the data **in place** instead of copying it out.
//!
//! One usage rule applies: **do not acquire a second guard on an object while
//! holding a write guard on that same object** (and do not `write` an object
//! while holding any guard on it) — the lock is not reentrant. Distinct
//! objects are fully independent.
//!
//! # Example
//!
//! ```
//! use pyrucast::handle::Handle;
//!
//! #[derive(Debug, PartialEq)]
//! struct Points(Vec<f64>);
//!
//! let h = Handle::new(Points(vec![1.0, 2.0]));
//! h.write().0.push(3.0);
//! assert_eq!(h.read().0.len(), 3);
//!
//! // A clone names the same object.
//! let g = h.clone();
//! assert!(g.same_object(&h));
//! ```

use parking_lot::lock_api::{ArcRwLockReadGuard, ArcRwLockWriteGuard};
use parking_lot::{RawRwLock, RwLock};
use std::fmt;
use std::sync::Arc;

/// Shared (read) access to an object, returned by [`Handle::read`].
///
/// Owned (`'static`): it can be returned from a function and stored in a
/// struct. Dereferences to `&T`; the lock is released when the guard goes out
/// of scope. The guard holds the object itself alive, so it may outlive every
/// handle it came from.
pub type ReadGuard<T> = ArcRwLockReadGuard<RawRwLock, T>;

/// Exclusive (write) access to an object, returned by [`Handle::write`].
///
/// Same ownership properties as [`ReadGuard`]; dereferences to `&mut T`.
pub type WriteGuard<T> = ArcRwLockWriteGuard<RawRwLock, T>;

/// A counted reference to an object, with its own lock.
///
/// Cloning shares the object rather than copying it — the whole point, since a
/// `SubMesh` or a `Coords` is large and shared by construction. Two handles
/// name the same object exactly when [`same_object`](Self::same_object) says
/// so.
// ANCHOR: declaration
pub struct Handle<T> {
    cell: Arc<RwLock<T>>,
}
// ANCHOR_END: declaration

impl<T> Handle<T> {
    /// Take ownership of `value` and hand back the first reference to it.
    ///
    /// **The single funnel through which objects are created.** Keeping it that
    /// way is what would make a future registry — for enumeration, or a
    /// whole-session save — a change to one function rather than a hunt through
    /// every construction site.
    pub fn new(value: T) -> Self {
        Self {
            cell: Arc::new(RwLock::new(value)),
        }
    }

    /// Shared access. Blocks while a writer holds the object.
    ///
    /// Infallible: holding `self` guarantees the object is there.
    pub fn read(&self) -> ReadGuard<T> {
        self.cell.read_arc()
    }

    /// Exclusive access. Blocks while any other guard holds the object.
    ///
    /// Infallible, for the same reason as [`read`](Self::read).
    pub fn write(&self) -> WriteGuard<T> {
        self.cell.write_arc()
    }

    /// Whether two handles name the **same object**.
    ///
    /// Identity, not equality: this is what lets an aggregate's union skip a
    /// sub it already holds, and what tells a field that its support is the
    /// mesh it was handed.
    pub fn same_object(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cell, &other.cell)
    }

    /// An opaque identity, hashable and comparable — for use as a map key when
    /// grouping by object.
    ///
    /// Two handles give the same `id` exactly when they name the same object.
    /// Like the tag shown by [`fmt::Display`], it is an **address**: unique
    /// among objects alive at the same instant, and reused once an object is
    /// gone. Safe to key a table built and consumed within one operation; not a
    /// durable identity to store or write to a file.
    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.cell) as *const () as usize
    }

    /// The object's address, as the short tag shown by [`fmt::Display`].
    ///
    /// Distinguishes objects **alive at the same instant** — which is what a
    /// human comparing two lines of output needs. It is *not* a durable
    /// identity: an address is reused once the object is gone, so two entries
    /// bearing the same tag in a log written over time may be two different
    /// objects.
    fn tag(&self) -> usize {
        self.id() & 0xff_ffff
    }
}

impl<T> Clone for Handle<T> {
    /// Written out rather than derived: `#[derive(Clone)]` would demand
    /// `T: Clone`, when cloning a handle never touches the object.
    fn clone(&self) -> Self {
        Self {
            cell: self.cell.clone(),
        }
    }
}

/// Short, lock-free view: `<SubMesh #7f3a2c>`.
///
/// Deliberately does **not** read the object. Printing a handle must stay free
/// — a `SubMesh` holds a connectivity of millions of entries, and a handle may
/// well be formatted while a write guard is held on it, which reading would
/// deadlock.
impl<T> fmt::Display for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let short = std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap_or("?");
        write!(f, "<{} #{:x}>", short, self.tag())
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Handle<{}> {{ #{:x} }}",
            std::any::type_name::<T>(),
            self.tag()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, PartialEq)]
    struct P(Vec<f64>);

    #[test]
    fn new_then_read() {
        let h = Handle::new(P(vec![1.0, 2.0]));
        assert_eq!(h.read().0, vec![1.0, 2.0]);
    }

    #[test]
    fn write_modifies_in_place() {
        let h = Handle::new(P(vec![]));
        h.write().0.push(7.0);
        assert_eq!(h.read().0, vec![7.0]);
    }

    #[test]
    fn clone_shares_the_object() {
        let a = Handle::new(P(vec![1.0]));
        let b = a.clone();
        assert!(a.same_object(&b));
        b.write().0.push(2.0);
        assert_eq!(a.read().0.len(), 2);
    }

    #[test]
    fn distinct_objects_are_not_the_same() {
        let a = Handle::new(P(vec![1.0]));
        let b = Handle::new(P(vec![1.0]));
        assert!(!a.same_object(&b), "equal contents are not one object");
    }

    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    struct Counted;
    impl Drop for Counted {
        fn drop(&mut self) {
            DROPPED.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// The lifetime contract: the object dies with the **last** handle, and
    /// dies once. This is what replaced the hand-written refcount, so it is the
    /// assertion that says the replacement behaves.
    #[test]
    fn the_object_dies_with_the_last_handle_and_only_then() {
        DROPPED.store(0, Ordering::SeqCst);
        let a = Handle::new(Counted);
        let b = a.clone();
        drop(a);
        assert_eq!(
            DROPPED.load(Ordering::SeqCst),
            0,
            "one handle still names it"
        );
        drop(b);
        assert_eq!(DROPPED.load(Ordering::SeqCst), 1, "exactly one drop");
    }

    /// A guard keeps the object alive on its own — an operator may hold data in
    /// place after the handle it came from is gone.
    #[test]
    fn a_guard_outlives_the_last_handle() {
        DROPPED.store(0, Ordering::SeqCst);
        let h = Handle::new(Counted);
        let g = h.read();
        drop(h);
        assert_eq!(
            DROPPED.load(Ordering::SeqCst),
            0,
            "the guard still holds it"
        );
        drop(g);
        assert_eq!(DROPPED.load(Ordering::SeqCst), 1);
    }

    /// Formatting must neither read the object nor block: here it happens while
    /// a write guard is held, which reading would deadlock.
    #[test]
    fn display_does_not_touch_the_object() {
        let h = Handle::new(P(vec![1.0]));
        let _w = h.write();
        let shown = format!("{h}");
        assert!(shown.starts_with("<P #"), "unexpected: {shown}");
    }

    #[test]
    fn concurrent_readers() {
        let h = Handle::new(P(vec![1.0, 2.0]));
        std::thread::scope(|s| {
            for _ in 0..4 {
                let h = h.clone();
                s.spawn(move || {
                    let g = h.read();
                    assert_eq!(g.0.len(), 2);
                });
            }
        });
    }
}
