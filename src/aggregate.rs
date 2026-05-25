//! Common trait for the store's "parent" containers (Mesh, FiniteElementSpace,
//! Model). Each is essentially a `Vec<Handle<Sub>>` and exposes the same
//! access grammar (`len`, `get`, indexing, iteration).
//!
//! Goals:
//! - factor out the access mechanics (zero duplicated `__getitem__`);
//! - guarantee uniform access: a sub-mesh from a mesh, a sub-space from an
//!   FE-space, or a sub-model from a model are iterated and indexed in
//!   strictly the same way.
//!
//! # Example
//!
//! ```
//! use pyrucast::aggregate::Aggregate;
//! use pyrucast::store::{insert, with, Handle};
//!
//! #[derive(serde::Serialize, serde::Deserialize)]
//! struct Item(u32);
//!
//! #[derive(Default)]
//! struct Bag { items: Vec<Handle<Item>> }
//!
//! impl Aggregate for Bag {
//!     type Sub = Item;
//!     fn items(&self) -> &[Handle<Item>] { &self.items }
//!     fn items_mut(&mut self) -> &mut Vec<Handle<Item>> { &mut self.items }
//! }
//!
//! let mut b = Bag::default();
//! assert!(b.is_empty());
//! b.push(insert(Item(42)));
//! assert_eq!(b.len(), 1);
//! let h = b.get(0).unwrap();
//! with(&h, |it| assert_eq!(it.0, 42)).unwrap();
//! ```

use crate::error::{PyrucastError, Result};
use crate::persist::Persist;
use crate::store::Handle;
use std::any::Any;

/// Typed container of objects held in the global store.
///
/// All access mechanics (length, indexing, iteration) are derived from the
/// two required methods [`Aggregate::items`] and [`Aggregate::items_mut`].
pub trait Aggregate {
    /// Type of the sub-object held in the store (referenced via `Handle<Sub>`).
    type Sub: Persist + Any + Send;

    /// Reference to the internal list of handles.
    fn items(&self) -> &[Handle<Self::Sub>];

    /// Mutable reference to the internal list of handles.
    fn items_mut(&mut self) -> &mut Vec<Handle<Self::Sub>>;

    fn len(&self) -> usize {
        self.items().len()
    }

    fn is_empty(&self) -> bool {
        self.items().is_empty()
    }

    /// Clone of the `i`-th handle. Explicit error if `i` is out of bounds.
    fn get(&self, i: usize) -> Result<Handle<Self::Sub>> {
        self.items().get(i).cloned().ok_or_else(|| {
            PyrucastError::Message(format!(
                "index {} out of bounds (len={})",
                i,
                self.len()
            ))
        })
    }

    /// Append a handle at the tail. Any extra validation (Configuration
    /// compatibility, duplicates, ...) is the caller's responsibility.
    fn push(&mut self, h: Handle<Self::Sub>) {
        self.items_mut().push(h);
    }

    /// Iterator over the internal handles.
    fn iter(&self) -> std::slice::Iter<'_, Handle<Self::Sub>> {
        self.items().iter()
    }

    /// Append all handles from `other` into `self` (clone each handle).
    fn extend_from(&mut self, other: &Self) {
        for h in other.iter() {
            self.items_mut().push(h.clone());
        }
    }

    /// Hook called by [`try_extend_from`] before concatenating. Override to
    /// enforce domain-specific constraints (e.g. same `Configuration`).
    /// The default accepts everything.
    fn check_merge_compatibility(&self, _other: &Self) -> crate::error::Result<()> {
        Ok(())
    }

    /// Check compatibility then append all handles from `other` into `self`.
    fn try_extend_from(&mut self, other: &Self) -> crate::error::Result<()> {
        self.check_merge_compatibility(other)?;
        self.extend_from(other);
        Ok(())
    }
}

// ─── Helper: Python indexing (signed → unsigned) ────────────────────────────

/// Normalize a Python index (possibly negative) against `len`.
///
/// Returns `Some(positive_idx)` if valid, `None` otherwise. The call site
/// turns `None` into the appropriate error (`PyIndexError` on the Python
/// side, `PyrucastError::Message` on the Rust side).
pub fn normalize_index(idx: isize, len: usize) -> Option<usize> {
    let n = len as isize;
    let i = if idx < 0 { n + idx } else { idx };
    if i < 0 || i >= n {
        None
    } else {
        Some(i as usize)
    }
}

// ─── Python macro: uniform __len__ / __getitem__ ────────────────────────────

/// Defines `__len__` and `__getitem__` on a `pyclass` that owns an `inner`
/// field implementing [`Aggregate`], wrapping each sub-handle in a
/// `$PySub { handle: h }` wrapper.
///
/// Python iteration (`for x in parent:`) naturally falls back to the
/// sequence protocol: if `__iter__` is not defined, CPython falls back to
/// `__getitem__` until it receives an `IndexError`.
///
/// # Usage example (post-refactor B)
///
/// ```ignore
/// pyrucast::impl_aggregate_pymethods!(PyMesh, PySubMesh, "Mesh");
/// pyrucast::impl_aggregate_pymethods!(PyFiniteElementSpace, PySubFiniteElementSpace, "FiniteElementSpace");
/// pyrucast::impl_aggregate_pymethods!(PyModel, PySubModel, "Model");
/// ```
#[cfg(feature = "python-api")]
#[macro_export]
macro_rules! impl_aggregate_pymethods {
    ($PyParent:ident, $PySub:ident, $name:literal) => {
        #[cfg_attr(
            feature = "stub-gen",
            pyo3_stub_gen::derive::gen_stub_pymethods
        )]
        #[pyo3::pymethods]
        impl $PyParent {
            fn __len__(&self) -> pyo3::PyResult<usize> {
                Ok($crate::aggregate::Aggregate::len(&self.inner))
            }

            fn __getitem__(&self, idx: isize) -> pyo3::PyResult<$PySub> {
                let n = $crate::aggregate::Aggregate::len(&self.inner);
                let i = $crate::aggregate::normalize_index(idx, n).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err(format!(
                        concat!($name, " index {} out of range (len={})"),
                        idx, n
                    ))
                })?;
                let h = $crate::aggregate::Aggregate::get(&self.inner, i)?;
                Ok($PySub { handle: h })
            }
        }
    };
}

// ─── Tests ──────────────────────────────────────────────────────────────────


#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{insert, with, Handle};
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct Item(u32);

    #[derive(Default)]
    struct Bag {
        items: Vec<Handle<Item>>,
    }
    impl Aggregate for Bag {
        type Sub = Item;
        fn items(&self) -> &[Handle<Item>] {
            &self.items
        }
        fn items_mut(&mut self) -> &mut Vec<Handle<Item>> {
            &mut self.items
        }
    }

    #[test]
    fn empty_bag_is_empty() {
        let b = Bag::default();
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
        assert!(b.get(0).is_err());
    }

    #[test]
    fn push_then_get() {
        let mut b = Bag::default();
        b.push(insert(Item(1)));
        b.push(insert(Item(2)));
        b.push(insert(Item(3)));
        assert_eq!(b.len(), 3);
        let h = b.get(1).unwrap();
        with(&h, |it| assert_eq!(it.0, 2)).unwrap();
    }

    #[test]
    fn iter_walks_in_order() {
        let mut b = Bag::default();
        b.push(insert(Item(10)));
        b.push(insert(Item(20)));
        let collected: Vec<u32> = b.iter().map(|h| with(h, |it| it.0).unwrap()).collect();
        assert_eq!(collected, vec![10, 20]);
    }

    #[test]
    fn out_of_bounds_message_includes_len() {
        let b = Bag::default();
        let err = b.get(5).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("out of bounds"));
        assert!(msg.contains("len=0"));
    }

    #[test]
    fn normalize_negative_index() {
        assert_eq!(normalize_index(0, 3), Some(0));
        assert_eq!(normalize_index(2, 3), Some(2));
        assert_eq!(normalize_index(-1, 3), Some(2));
        assert_eq!(normalize_index(-3, 3), Some(0));
        assert_eq!(normalize_index(-4, 3), None);
        assert_eq!(normalize_index(3, 3), None);
        assert_eq!(normalize_index(0, 0), None);
    }
}
