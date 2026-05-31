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
//!     fn type_name() -> &'static str { "Bag" }
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
pub trait Aggregate: Default {
    /// Type of the sub-object held in the store (referenced via `Handle<Sub>`).
    type Sub: Persist + Any + Send;

    /// Reference to the internal list of handles.
    fn items(&self) -> &[Handle<Self::Sub>];

    /// Mutable reference to the internal list of handles.
    fn items_mut(&mut self) -> &mut Vec<Handle<Self::Sub>>;

    /// Human-readable name of this aggregate type (e.g. `"Mesh"`).
    /// Used by the default `Debug` and `Display` implementations.
    fn type_name() -> &'static str;

    /// Construct an empty aggregate (no sub-items). Equivalent to `Default::default()`.
    fn empty() -> Self where Self: Sized { Self::default() }

    /// Plural label for the sub-item used by the default `Display`
    /// (e.g. `"submesh(es)"`). Defaults to `"item(s)"`.
    fn sub_display_name() -> &'static str { "item(s)" }

    /// Optional suffix appended by the default `Display` after the count
    /// (e.g. `", 12 cell(s) total"`). Defaults to nothing.
    fn display_extra(&self) -> Option<String> { None }

    /// Merge `self` and `other` into a fresh aggregate.
    ///
    /// Delegates to [`try_extend_from`] so domain constraints (e.g.
    /// `Configuration` compatibility for `Mesh`) are enforced.
    fn merge(&self, other: &Self) -> Result<Self> where Self: Sized {
        let mut result = Self::default();
        result.try_extend_from(self)?;
        result.try_extend_from(other)?;
        Ok(result)
    }

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

    /// Hook called before inserting a single handle. Override to enforce
    /// domain-specific constraints (e.g. same `Configuration` for `Mesh`).
    /// The default accepts everything.
    fn check_push(&self, _h: &Handle<Self::Sub>) -> crate::error::Result<()> {
        Ok(())
    }

    /// Hook called after a handle has been successfully pushed. Override to
    /// perform side-effects such as cache invalidation.
    /// The default is a no-op.
    fn post_push(&mut self) {}

    /// Check compatibility (via [`check_push`]) then append a single handle.
    fn add_sub(&mut self, h: Handle<Self::Sub>) -> crate::error::Result<()> {
        self.check_push(&h)?;
        self.push(h);
        self.post_push();
        Ok(())
    }

    /// Check compatibility (via [`check_push`] on the first item of `other`)
    /// then append all handles from `other` into `self`.
    fn try_extend_from(&mut self, other: &Self) -> crate::error::Result<()> {
        if let Some(h) = other.items().first() {
            self.check_push(h)?;
        }
        for h in other.iter() {
            self.items_mut().push(h.clone());
            self.post_push();
        }
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
/// `$Sub { handle: h }` wrapper.
///
/// Python iteration (`for x in parent:`) naturally falls back to the
/// sequence protocol: if `__iter__` is not defined, CPython falls back to
/// `__getitem__` until it receives an `IndexError`.
///
/// # Struct precondition
///
/// The struct `$T` **must** have a field named `inner` that implements
/// [`Aggregate`]. The name `inner` is hardcoded by this macro.
///
/// # Usage
///
/// ```ignore
/// pyrucast::impl_aggregate_pymethods!(PyMesh, PySubMesh, "Mesh", submesh);
/// pyrucast::impl_aggregate_pymethods!(PyFiniteElementSpace, PySubFiniteElementSpace, "FiniteElementSpace", subspace);
/// pyrucast::impl_aggregate_pymethods!(PyModel, PySubModel, "Model", sub_model);
/// ```
///
/// `$sub` is given once; `paste!` derives `{$sub}_count`, `$sub(i)`, and
/// `add_{$sub}` automatically.
#[cfg(feature = "python-api")]
#[macro_export]
macro_rules! impl_aggregate_pymethods {
    ($T:ident, $Sub:ident, $name:literal, $sub:ident) => {
        paste::paste! {
            #[cfg_attr(
                feature = "stub-gen",
                pyo3_stub_gen::derive::gen_stub_pymethods
            )]
            #[pyo3::pymethods]
            impl $T {
                fn __len__(&self) -> pyo3::PyResult<usize> {
                    Ok($crate::aggregate::Aggregate::len(&self.inner))
                }

                fn __getitem__(&self, idx: isize) -> pyo3::PyResult<$Sub> {
                    let n = $crate::aggregate::Aggregate::len(&self.inner);
                    let i = $crate::aggregate::normalize_index(idx, n).ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err(format!(
                            concat!($name, " index {} out of range (len={})"),
                            idx, n
                        ))
                    })?;
                    let h = $crate::aggregate::Aggregate::get(&self.inner, i)?;
                    Ok($Sub { handle: h })
                }

                fn [<$sub _count>](&self) -> pyo3::PyResult<usize> {
                    Ok($crate::aggregate::Aggregate::len(&self.inner))
                }

                fn $sub(&self, i: usize) -> pyo3::PyResult<$Sub> {
                    let h = $crate::aggregate::Aggregate::get(&self.inner, i)?;
                    Ok($Sub { handle: h })
                }

                fn add_sub(&mut self, sub: pyo3::PyRef<'_, $Sub>) -> pyo3::PyResult<()> {
                    $crate::aggregate::Aggregate::add_sub(&mut self.inner, sub.handle.clone())?;
                    Ok(())
                }

                fn [<add_ $sub>](&mut self, sub: pyo3::PyRef<'_, $Sub>) -> pyo3::PyResult<()> {
                    $crate::aggregate::Aggregate::add_sub(&mut self.inner, sub.handle.clone())?;
                    Ok(())
                }

                fn __repr__(&self) -> pyo3::PyResult<String> {
                    Ok(format!("{:?}", self.inner))
                }

                fn __str__(&self) -> pyo3::PyResult<String> {
                    Ok(format!("{}", self.inner))
                }
            }
        }
    };
}

// ─── Unified aggregate macro ────────────────────────────────────────────────

/// Derive a complete `Aggregate` implementation for a struct whose only
/// collection field is named `subs: Vec<Handle<Sub>>`.
///
/// # Struct precondition
///
/// The struct `$T` **must** have a field named `subs: Vec<Handle<$Sub>>`.
/// The name `subs` is hardcoded by this macro.
///
/// # Derived names
///
/// `$sub` is given once; `paste!` derives:
/// * `$sub(i)`             — indexed accessor
/// * `{$sub}_count()`     — length alias
/// * `add_{$sub}(h)`      — checked push
///
/// An optional trailing `{ … }` block is forwarded verbatim into the
/// `impl Aggregate` body to override default methods (`check_push`,
/// `display_extra`, …).
///
/// # Usage
/// ```ignore
/// impl_aggregate!(Mesh, SubMesh, submesh, "submesh(es)", {
///     fn check_push(&self, h: &Handle<SubMesh>) -> Result<()> { … }
///     fn display_extra(&self) -> Option<String> { … }
/// });
/// impl_aggregate!(Model, SubModel, sub_model, "sub-model(s)");
/// ```
#[macro_export]
macro_rules! impl_aggregate {
    // Without Aggregate overrides.
    ($T:ty, $Sub:ty, $sub:ident, $name:literal) => {
        $crate::impl_aggregate!($T, $Sub, $sub, $name, {});
    };
    // With Aggregate overrides (check_push, display_extra, …).
    ($T:ty, $Sub:ty, $sub:ident, $name:literal, { $($override:tt)* }) => {
        paste::paste! {
            impl $crate::aggregate::Aggregate for $T {
                type Sub = $Sub;
                fn items(&self) -> &[$crate::store::Handle<$Sub>] { &self.subs }
                fn items_mut(&mut self) -> &mut Vec<$crate::store::Handle<$Sub>> {
                    &mut self.subs
                }
                fn type_name() -> &'static str { stringify!($T) }
                fn sub_display_name() -> &'static str { $name }
                $($override)*
            }

            impl $T {
                pub fn [<$sub _count>](&self) -> usize {
                    $crate::aggregate::Aggregate::len(self)
                }
                pub fn $sub(&self, i: usize)
                    -> $crate::error::Result<$crate::store::Handle<$Sub>>
                {
                    $crate::aggregate::Aggregate::get(self, i)
                }
                pub fn [<add_ $sub>](
                    &mut self,
                    h: $crate::store::Handle<$Sub>,
                ) -> $crate::error::Result<()> {
                    $crate::aggregate::Aggregate::add_sub(self, h)
                }
            }

            $crate::impl_aggregate_std_traits!($T);
        }
    };
}

// ─── Std-trait macro ────────────────────────────────────────────────────────

/// Generate `Index`, `IntoIterator`, `fmt::Debug`, `fmt::Display`, and
/// `Add<&T> for &T` (returning `Result<T>`) for a concrete type that
/// implements [`Aggregate`].
///
/// # Usage
/// ```ignore
/// impl_aggregate_std_traits!(Mesh);
/// impl_aggregate_std_traits!(FiniteElementSpace);
/// ```
#[macro_export]
macro_rules! impl_aggregate_std_traits {
    ($T:ty) => {
        impl std::ops::Index<usize> for $T {
            type Output = $crate::store::Handle<<$T as $crate::aggregate::Aggregate>::Sub>;
            fn index(&self, idx: usize) -> &Self::Output {
                &$crate::aggregate::Aggregate::items(self)[idx]
            }
        }

        impl<'a> IntoIterator for &'a $T {
            type Item = &'a $crate::store::Handle<<$T as $crate::aggregate::Aggregate>::Sub>;
            type IntoIter = std::slice::Iter<'a, $crate::store::Handle<<$T as $crate::aggregate::Aggregate>::Sub>>;
            fn into_iter(self) -> Self::IntoIter {
                $crate::aggregate::Aggregate::iter(self)
            }
        }

        impl std::fmt::Debug for $T {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(<$T as $crate::aggregate::Aggregate>::type_name())
                    .field("count", &$crate::aggregate::Aggregate::len(self))
                    .field("items", &$crate::aggregate::Aggregate::items(self))
                    .finish()
            }
        }

        impl std::fmt::Display for $T {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "{}: {} {}",
                    <$T as $crate::aggregate::Aggregate>::type_name(),
                    $crate::aggregate::Aggregate::len(self),
                    <$T as $crate::aggregate::Aggregate>::sub_display_name(),
                )?;
                if let Some(extra) = $crate::aggregate::Aggregate::display_extra(self) {
                    f.write_str(&extra)?;
                }
                Ok(())
            }
        }

        impl std::ops::Add<&$T> for &$T {
            type Output = $crate::error::Result<$T>;
            fn add(self, rhs: &$T) -> Self::Output {
                $crate::aggregate::Aggregate::merge(self, rhs)
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
        fn items(&self) -> &[Handle<Item>] { &self.items }
        fn items_mut(&mut self) -> &mut Vec<Handle<Item>> { &mut self.items }
        fn type_name() -> &'static str { "Bag" }
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
