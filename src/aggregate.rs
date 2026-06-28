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
//! use pyrucast::store::{insert, read, Handle};
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
//! assert_eq!(read(&h).unwrap().0, 42);
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
    type Sub: Persist + Any + Send + Sync;

    /// Reference to the internal list of handles.
    fn items(&self) -> &[Handle<Self::Sub>];

    /// Mutable reference to the internal list of handles.
    fn items_mut(&mut self) -> &mut Vec<Handle<Self::Sub>>;

    /// Human-readable name of this aggregate type (e.g. `"Mesh"`).
    /// Used by the default `Debug` and `Display` implementations.
    fn type_name() -> &'static str;

    /// Construct an empty aggregate (no sub-items). Equivalent to `Default::default()`.
    fn empty() -> Self
    where
        Self: Sized,
    {
        Self::default()
    }

    /// Plural label for the sub-item used by the default `Display`
    /// (e.g. `"submesh(es)"`). Defaults to `"item(s)"`.
    fn sub_display_name() -> &'static str {
        "item(s)"
    }

    /// Optional suffix appended by the default `Display` after the count
    /// (e.g. `", 12 cell(s) total"`). Defaults to nothing.
    fn display_extra(&self) -> Option<String> {
        None
    }

    /// Union of `self` and `other` into a fresh aggregate.
    ///
    /// Exposed to Rust as [`Aggregate::union`] and to Python as `a | b`.
    /// Sub-objects are deduplicated **by handle identity**
    /// ([`Handle::same_slot`]): a sub already present (same store slot) is
    /// not added twice. Order is first-seen. Domain constraints (e.g.
    /// `Coords` compatibility for `Mesh`) are enforced via
    /// [`Aggregate::try_extend_from`]. After the union, [`Aggregate::finalize`]
    /// runs — a no-op for most aggregates, but the fields override it to fuse
    /// zones sharing the same support.
    fn merge(&self, other: &Self) -> Result<Self>
    where
        Self: Sized,
    {
        let mut result = Self::default();
        result.try_extend_from(self)?;
        result.try_extend_from(other)?;
        result.finalize()?;
        Ok(result)
    }

    /// Union with another aggregate — the named Rust entry point for the
    /// composition operator (`a | b` in Python). Alias of
    /// [`Aggregate::merge`]; see it for the deduplication and finalization
    /// semantics.
    fn union(&self, other: &Self) -> Result<Self>
    where
        Self: Sized,
    {
        self.merge(other)
    }

    /// Union with a single sub-object `h`: a fresh aggregate holding
    /// `self`'s subs plus `h`, unless `h`'s slot is already present. The
    /// sub-handle is shared (refcount bump), then [`Aggregate::finalize`]
    /// runs. Python: `aggregate | sub`.
    fn union_sub(&self, h: &Handle<Self::Sub>) -> Result<Self>
    where
        Self: Sized,
    {
        let mut out = Self::default();
        out.try_extend_from(self)?;
        if !out.contains_handle(h) {
            out.add_sub(h.clone())?;
        }
        out.finalize()?;
        Ok(out)
    }

    /// Fresh aggregate holding the sub-objects at `indices` (already
    /// resolved against `len`, in the given order). Each handle is shared
    /// (refcount bump), not deep-copied. Backs Python slicing (`agg[i:j:k]`).
    ///
    /// Reuses [`Aggregate::add_sub`] — so [`Aggregate::check_push`] invariants
    /// (e.g. `Coords` compatibility for `Mesh`) still hold — then runs
    /// [`Aggregate::finalize`]. A slice yields distinct indices, so no handle
    /// is added twice; on an already-finalized source `finalize` is a no-op.
    fn subset<I: IntoIterator<Item = usize>>(&self, indices: I) -> Result<Self>
    where
        Self: Sized,
    {
        let mut out = Self::default();
        for i in indices {
            out.add_sub(self.get(i)?)?;
        }
        out.finalize()?;
        Ok(out)
    }

    /// Hook called at the end of [`Aggregate::merge`], once the union by
    /// handle is complete. Override to fuse/normalize sub-objects beyond
    /// handle identity (e.g. fields fusing two zones on the same support).
    /// The default is a no-op.
    fn finalize(&mut self) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }

    fn len(&self) -> usize {
        self.items().len()
    }

    fn is_empty(&self) -> bool {
        self.items().is_empty()
    }

    /// The sole sub-handle of a **unitary** aggregate (exactly one sub).
    ///
    /// This is the parent→sub coercion primitive (see `CONVENTIONS.md`,
    /// « Agrégats : un ou plusieurs ») used at the few boundaries that
    /// genuinely need a single sub — e.g. a `SubNodeField` support accepting
    /// a unitary `Mesh`. Errors with a clear message if the aggregate holds
    /// zero or more than one sub.
    fn unit(&self) -> Result<Handle<Self::Sub>> {
        match self.items() {
            [h] => Ok(h.clone()),
            items => Err(PyrucastError::Message(format!(
                "expected a unitary {} (exactly one {}), found {}",
                Self::type_name(),
                Self::sub_display_name(),
                items.len(),
            ))),
        }
    }

    /// Clone of the `i`-th handle. Explicit error if `i` is out of bounds.
    fn get(&self, i: usize) -> Result<Handle<Self::Sub>> {
        self.items().get(i).cloned().ok_or_else(|| {
            PyrucastError::Message(format!("index {} out of bounds (len={})", i, self.len()))
        })
    }

    /// Append a handle at the tail. Any extra validation (Coords
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

    /// Whether a sub with the same store slot as `h` is already held
    /// ([`Handle::same_slot`]). Basis of the union's deduplication.
    fn contains_handle(&self, h: &Handle<Self::Sub>) -> bool {
        self.items().iter().any(|existing| existing.same_slot(h))
    }

    /// Hook called before inserting a single handle. Override to enforce
    /// domain-specific constraints (e.g. same `Coords` for `Mesh`).
    /// The default accepts everything.
    fn check_push(&self, _h: &Handle<Self::Sub>) -> crate::error::Result<()> {
        Ok(())
    }

    /// Hook called after a handle has been successfully pushed. Override to
    /// perform side-effects such as cache invalidation.
    /// The default is a no-op.
    fn post_push(&mut self) {}

    /// Check compatibility (via [`Aggregate::check_push`]) then append a single handle.
    fn add_sub(&mut self, h: Handle<Self::Sub>) -> crate::error::Result<()> {
        self.check_push(&h)?;
        self.push(h);
        self.post_push();
        Ok(())
    }

    /// Check compatibility (via [`Aggregate::check_push`] on the first item of `other`)
    /// then append handles from `other` into `self`, **deduplicating by
    /// handle identity** ([`Handle::same_slot`]): a sub already present is
    /// skipped. This is the union semantics of `|`.
    fn try_extend_from(&mut self, other: &Self) -> crate::error::Result<()> {
        if let Some(h) = other.items().first() {
            self.check_push(h)?;
        }
        for h in other.iter() {
            if self.contains_handle(h) {
                continue;
            }
            self.items_mut().push(h.clone());
            self.post_push();
        }
        Ok(())
    }
}

// ─── Debug helper: recurse into each sub-object ─────────────────────────────

/// Wraps an aggregate's handle slice so its `Debug` **dereferences** each
/// handle and prints the full `Debug` of the pointed-to sub-object, instead
/// of only the handle's `idx`/`gen`. A handle that can no longer be resolved
/// (stale / collected slot) falls back to printing the handle itself.
///
/// Honours the alternate (`{:#?}`) flag, so the sub-objects are
/// pretty-printed and indented when the aggregate is.
///
/// Safe against the per-type, non-reentrant store mutex: the only lock taken
/// is the sub-object store's, and every sub-object `Debug` is itself
/// lock-free, so no nesting of `with::<Sub>` inside `with::<Sub>` occurs.
pub struct DebugItems<'a, S: Persist + Any + Send + Sync>(pub &'a [Handle<S>]);

impl<S: Persist + Any + Send + Sync + std::fmt::Debug> std::fmt::Debug for DebugItems<'_, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for h in self.0 {
            list.entry(&DebugItem(h));
        }
        list.finish()
    }
}

/// A single dereferenced handle (see [`DebugItems`]).
struct DebugItem<'a, S: Persist + Any + Send + Sync>(&'a Handle<S>);

impl<S: Persist + Any + Send + Sync + std::fmt::Debug> std::fmt::Debug for DebugItem<'_, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match crate::store::read(self.0) {
            Ok(s) => {
                if f.alternate() {
                    write!(f, "{:#?}", &*s)
                } else {
                    write!(f, "{:?}", &*s)
                }
            }
            // Unresolvable handle: fall back to its idx/gen identity.
            Err(_) => std::fmt::Debug::fmt(self.0, f),
        }
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
/// `$Inner` is the **Rust** aggregate type stored in `$T`'s `inner` field
/// (e.g. `Mesh` behind `PyMesh`); it owns the `union`/`union_sub`/`union_subs`
/// constructors that back the `|` operator. The macro wires both the
/// aggregate-level `|` (on `$T`) **and** the sub-level `|` (on `$Sub`), so a
/// single invocation gives the whole uniform union surface.
///
/// # Usage
///
/// ```ignore
/// pyrucast::impl_aggregate_pymethods!(PyMesh, PySubMesh, "Mesh", submesh, Mesh);
/// pyrucast::impl_aggregate_pymethods!(PyModel, PySubModel, "Model", sub_model, Model);
/// ```
///
/// `$sub` is given once; `paste!` derives `{$sub}_count`, `$sub(i)`, and
/// `add_{$sub}` automatically.
#[cfg(feature = "python-api")]
#[macro_export]
macro_rules! impl_aggregate_pymethods {
    ($T:ident, $Sub:ident, $name:literal, $sub:ident, $Inner:ty) => {
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

                /// `agg[i]` → the typed **view** of zone `i` (a `$Sub`,
                /// negative indices supported); `agg[i:j:k]` → a **fresh
                /// aggregate** of the same type holding the sliced zones
                /// (Python slicing: step, negative bounds). Other key types
                /// raise `TypeError`.
                fn __getitem__(
                    &self,
                    py: pyo3::Python<'_>,
                    key: &pyo3::Bound<'_, pyo3::PyAny>,
                ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                    let n = $crate::aggregate::Aggregate::len(&self.inner);

                    // Slice: agg[1:3], agg[::2], agg[-2:] ...
                    if let Ok(slice) = key.cast::<pyo3::types::PySlice>() {
                        let ind = slice.indices(n as isize)?;
                        let mut idxs = Vec::new();
                        let mut i = ind.start;
                        if ind.step > 0 {
                            while i < ind.stop {
                                idxs.push(i as usize);
                                i += ind.step;
                            }
                        } else {
                            while i > ind.stop {
                                idxs.push(i as usize);
                                i += ind.step;
                            }
                        }
                        let inner = $crate::aggregate::Aggregate::subset(&self.inner, idxs)?;
                        return Ok(pyo3::Py::new(py, $T { inner })?.into_any());
                    }

                    // Integer: agg[0], agg[-1]
                    let idx: isize = key.extract().map_err(|_| {
                        pyo3::exceptions::PyTypeError::new_err(concat!(
                            $name,
                            " indices must be integers or slices"
                        ))
                    })?;
                    let i = $crate::aggregate::normalize_index(idx, n).ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err(format!(
                            concat!($name, " index {} out of range (len={})"),
                            idx, n
                        ))
                    })?;
                    let h = $crate::aggregate::Aggregate::get(&self.inner, i)?;
                    Ok(pyo3::Py::new(py, $Sub { handle: h })?.into_any())
                }

                /// The sole sub-object **view** of a unitary aggregate
                /// (exactly one sub), else a clear error. Use it where the
                /// single-zone case needs a sub method: `parent.unit().m(...)`.
                /// More honest than `parent[0]` (which silently takes the
                /// first of several) — see `CONVENTIONS.md`.
                fn unit(&self) -> pyo3::PyResult<$Sub> {
                    let h = $crate::aggregate::Aggregate::unit(&self.inner)?;
                    Ok($Sub { handle: h })
                }

                fn add_sub(&mut self, sub: pyo3::PyRef<'_, $Sub>) -> pyo3::PyResult<()> {
                    $crate::aggregate::Aggregate::add_sub(&mut self.inner, sub.handle.clone())?;
                    Ok(())
                }

                fn __repr__(&self) -> pyo3::PyResult<String> {
                    Ok(format!("{:?}", self.inner))
                }

                fn __str__(&self) -> pyo3::PyResult<String> {
                    Ok(format!("{}", self.inner))
                }

                /// Print the full content (third display level) to stdout:
                /// every sub-object's values/topology, beyond `repr`'s bounded
                /// structure. Returns nothing.
                #[pyo3(signature = (precision=3, max_rows=20, max_cols=12))]
                fn dump(
                    &self,
                    py: pyo3::Python<'_>,
                    precision: usize,
                    max_rows: usize,
                    max_cols: usize,
                ) -> pyo3::PyResult<()> {
                    let text = $crate::dump::Dump::render(
                        &self.inner,
                        &$crate::dump::DumpOptions { precision, max_rows, max_cols },
                    );
                    $crate::dump::py_print(py, &text)
                }

                /// `a | b` — **union** of this aggregate with `other`. `other`
                /// may be another aggregate of the same type or a single
                /// sub-object. Sub-objects already present (same store slot)
                /// are not added twice; remaining handles are **shared**
                /// (refcount bump), not deep-copied. The fields additionally
                /// fuse zones sharing a support (see their `finalize`), so
                /// `|` may raise on incoherent values. Returns
                /// `NotImplemented` for any other type so Python can fall
                /// back to the right operand's `__ror__`.
                fn __or__(
                    &self,
                    py: pyo3::Python<'_>,
                    other: &pyo3::Bound<'_, pyo3::PyAny>,
                ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                    use $crate::aggregate::Aggregate;
                    if let Ok(o) = other.extract::<pyo3::PyRef<'_, $T>>() {
                        let inner = self.inner.union(&o.inner)?;
                        return Ok(pyo3::Py::new(py, $T { inner })?.into_any());
                    }
                    if let Ok(s) = other.extract::<pyo3::PyRef<'_, $Sub>>() {
                        let inner = self.inner.union_sub(&s.handle)?;
                        return Ok(pyo3::Py::new(py, $T { inner })?.into_any());
                    }
                    Ok(py.NotImplemented())
                }
            }

            // Sub-level `|` (uniform with the aggregate-level one above):
            // `sub | sub` → a fresh aggregate holding both sub-objects.
            #[cfg_attr(
                feature = "stub-gen",
                pyo3_stub_gen::derive::gen_stub_pymethods
            )]
            #[pyo3::pymethods]
            impl $Sub {
                /// `sub | sub` → a fresh aggregate holding both sub-objects
                /// (first-seen order, deduplicated by handle, then finalized).
                /// Sub-handles are shared (refcount bump). Returns
                /// `NotImplemented` for any other right-hand type.
                fn __or__(
                    &self,
                    py: pyo3::Python<'_>,
                    other: &pyo3::Bound<'_, pyo3::PyAny>,
                ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
                    if let Ok(o) = other.extract::<pyo3::PyRef<'_, $Sub>>() {
                        let inner = <$Inner>::union_subs(&self.handle, &o.handle)?;
                        return Ok(pyo3::Py::new(py, $T { inner })?.into_any());
                    }
                    Ok(py.NotImplemented())
                }
            }
        }
    };
}

// ─── Python dump method (non-aggregate wrappers) ────────────────────────────

/// Define the `dump(precision, max_rows, max_cols) -> str` pymethod on a
/// `pyclass` wrapper whose payload implements [`crate::dump::Dump`].
///
/// Two forms depending on how the wrapper holds its payload:
/// * `handle $PyT, $field` — `$field: Handle<Sub>`, resolved through the store;
/// * `value  $PyT, $field` — `$field` is an owned value (e.g. a view).
///
/// Aggregate wrappers get `dump` from [`impl_aggregate_pymethods`] instead.
#[cfg(feature = "python-api")]
#[macro_export]
macro_rules! impl_dump_pymethod {
    (handle $PyT:ty, $field:ident) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pyo3::pymethods]
        impl $PyT {
            /// Print the full content (third display level) to stdout: values /
            /// topology, beyond `repr`'s bounded structure. Returns nothing.
            #[pyo3(signature = (precision=3, max_rows=20, max_cols=12))]
            fn dump(
                &self,
                py: pyo3::Python<'_>,
                precision: usize,
                max_rows: usize,
                max_cols: usize,
            ) -> pyo3::PyResult<()> {
                let opts = $crate::dump::DumpOptions {
                    precision,
                    max_rows,
                    max_cols,
                };
                let text = $crate::dump::Dump::render(&*$crate::store::read(&self.$field)?, &opts);
                $crate::dump::py_print(py, &text)
            }
        }
    };
    (value $PyT:ty, $field:ident) => {
        #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[pyo3::pymethods]
        impl $PyT {
            /// Print the full content (third display level) to stdout: values /
            /// topology, beyond `repr`'s bounded structure. Returns nothing.
            #[pyo3(signature = (precision=3, max_rows=20, max_cols=12))]
            fn dump(
                &self,
                py: pyo3::Python<'_>,
                precision: usize,
                max_rows: usize,
                max_cols: usize,
            ) -> pyo3::PyResult<()> {
                let opts = $crate::dump::DumpOptions {
                    precision,
                    max_rows,
                    max_cols,
                };
                let text = $crate::dump::Dump::render(&self.$field, &opts);
                $crate::dump::py_print(py, &text)
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
/// Access goes through the [`Aggregate`] trait (generic, no per-type
/// aliases): `len()`, `is_empty()`, `get(i)`, `add_sub(h)`, `iter()`,
/// `unit()`, `union()`/`union_sub()`, plus `Index`/`IntoIterator` and the
/// `union_subs(a, b)` constructor from [`crate::impl_aggregate_std_traits`].
/// The `$sub` snake-name is still passed (it names the sub in docs/call
/// sites) but no longer generates methods.
///
/// An optional trailing `{ … }` block is forwarded verbatim into the
/// `impl Aggregate` body to override default methods (`check_push`,
/// `display_extra`, …).
///
/// # Usage
/// ```text
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

            $crate::impl_aggregate_std_traits!($T);
        }
    };
}

// ─── Std-trait macro ────────────────────────────────────────────────────────

/// Generate `Index`, `IntoIterator`, `fmt::Debug`, `fmt::Display`, and the
/// `T::union_subs(a, b)` constructor (`sub | sub`, returning `Result<T>`)
/// for a concrete type that implements [`Aggregate`].
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
            type IntoIter = std::slice::Iter<
                'a,
                $crate::store::Handle<<$T as $crate::aggregate::Aggregate>::Sub>,
            >;
            fn into_iter(self) -> Self::IntoIter {
                $crate::aggregate::Aggregate::iter(self)
            }
        }

        impl std::fmt::Debug for $T {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(<$T as $crate::aggregate::Aggregate>::type_name())
                    .field("count", &$crate::aggregate::Aggregate::len(self))
                    .field(
                        "items",
                        &$crate::aggregate::DebugItems($crate::aggregate::Aggregate::items(self)),
                    )
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

        // `union` / `union_sub` (aggregate | aggregate, aggregate | sub)
        // are default methods on the `Aggregate` trait. Only `sub | sub`
        // needs a generated builder here: a sub-handle alone cannot name
        // its parent aggregate type, so we hang the constructor on `$T`.
        impl $T {
            /// Build the aggregate that is the union of two sub-objects
            /// (first-seen order, deduplicated by handle, then finalized).
            /// Backs Python's `sub | sub`.
            pub fn union_subs(
                a: &$crate::store::Handle<<$T as $crate::aggregate::Aggregate>::Sub>,
                b: &$crate::store::Handle<<$T as $crate::aggregate::Aggregate>::Sub>,
            ) -> $crate::error::Result<$T> {
                use $crate::aggregate::Aggregate;
                let mut out = <$T as ::std::default::Default>::default();
                out.add_sub(a.clone())?;
                if !a.same_slot(b) {
                    out.add_sub(b.clone())?;
                }
                out.finalize()?;
                Ok(out)
            }
        }
    };
}

// ─── Aggregate Dump macro ────────────────────────────────────────────────────

/// Derive [`crate::dump::Dump`] for an [`Aggregate`] whose `Sub` is itself
/// `Dump`: the one-line `Display` summary, then each sub-object's full `dump`
/// indented under a `── [i] ──` marker.
///
/// Kept separate from [`crate::impl_aggregate_std_traits`] so aggregates with a
/// non-generic content layout (e.g. `Matrix`, dumped as a single global grid)
/// can hand-write their own `Dump` instead.
///
/// # Usage
/// ```ignore
/// impl_aggregate_dump!(Mesh);
/// ```
#[macro_export]
macro_rules! impl_aggregate_dump {
    ($T:ty) => {
        impl $crate::dump::Dump for $T {
            fn render(&self, opts: &$crate::dump::DumpOptions) -> String {
                let mut out = format!("{self}\n");
                for (i, h) in $crate::aggregate::Aggregate::items(self).iter().enumerate() {
                    let body = $crate::store::read(h)
                        .map(|s| $crate::dump::Dump::render(&*s, opts))
                        .unwrap_or_else(|e| format!("<{e}>"));
                    out.push_str(&format!("── [{i}] ──\n"));
                    for line in body.lines() {
                        out.push_str("  ");
                        out.push_str(line);
                        out.push('\n');
                    }
                }
                out
            }
        }
    };
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{insert, read, Handle};
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
        fn type_name() -> &'static str {
            "Bag"
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
        assert_eq!(read(&h).unwrap().0, 2);
    }

    #[test]
    fn iter_walks_in_order() {
        let mut b = Bag::default();
        b.push(insert(Item(10)));
        b.push(insert(Item(20)));
        let collected: Vec<u32> = b.iter().map(|h| read(h).unwrap().0).collect();
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
    fn unit_returns_sole_handle_or_errors() {
        let mut b = Bag::default();
        assert!(b.unit().is_err()); // empty → error
        b.push(insert(Item(7)));
        let h = b.unit().unwrap();
        assert_eq!(read(&h).unwrap().0, 7);
        b.push(insert(Item(8)));
        assert!(b.unit().is_err()); // two → error
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
