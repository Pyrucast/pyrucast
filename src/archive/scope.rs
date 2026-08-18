//! The archive scope — where a [`Handle`] becomes a number, and back.
//!
//! A handle is an **address**. An address has no meaning in another process, so
//! it cannot be written to a file: what goes on disk is an identifier **local to
//! the file**, and the table that maps one to the other lives here.
//!
//! The table is installed for the duration of one [`save`](super::save) or
//! [`load`](super::load) and removed on the way out. Outside it, serializing a
//! handle is an error — never a panic, never a byte written at random.
//!
//! # Writing: the walk *is* the serialization
//!
//! Nothing declares which objects an object points at. [`intern_handle`] is
//! called by `Handle::serialize`, and it recurses:
//!
//! ```text
//! already seen ?  → write its identifier, done
//! otherwise       → reserve an identifier
//!                   serialize the pointed-to object into its own record
//!                      (which, recursively, discovers its own dependencies)
//!                   deposit the record
//!                   write the identifier
//! ```
//!
//! Two consequences fall out for free. Records land in **post-order**, so every
//! dependency precedes what references it and reading is a plain forward loop.
//! And a **cycle** is caught: an identifier that is reserved but not yet
//! deposited cannot be referenced, so the error names the object instead of
//! letting the stack overflow.
//!
//! # Reading
//!
//! Records are decoded in file order into a table `identifier → handle`. By the
//! time a record mentions an identifier, the object behind it is already there,
//! which is what the post-order buys.

use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;

use super::Archivable;

/// One object as it sits in the file: its identifier, its type tag, its bytes.
pub(crate) struct Record {
    pub id: u32,
    pub tag: &'static str,
    pub payload: Vec<u8>,
}

/// A record being built. `Reserved` is the window during which the object's own
/// dependencies are being walked — seeing it referenced then means a cycle.
enum Slot {
    Reserved(&'static str),
    Written(Record),
}

#[derive(Default)]
struct WriteScope {
    /// `Handle::id()` (an address) → identifier local to the file.
    seen: HashMap<usize, u32>,
    slots: Vec<Slot>,
    /// Identifiers in the order their record was **deposited**, which is the
    /// post-order — a parent reserves its identifier before walking into its
    /// children, so identifiers alone run the wrong way round.
    order: Vec<u32>,
}

#[derive(Default)]
struct ReadScope {
    /// Identifier → its tag and the object, type erased (a `Handle<T>`). The
    /// tag is kept because a *root* entry names an identifier, and wrapping it
    /// as a [`Root`](super::Root) needs to know which type it is.
    objects: HashMap<u32, (String, Box<dyn Any + Send + Sync>)>,
}

thread_local! {
    static WRITING: RefCell<Option<WriteScope>> = const { RefCell::new(None) };
    static READING: RefCell<Option<ReadScope>> = const { RefCell::new(None) };
}

/// Error raised when a handle is (de)serialized with no archive around it.
fn no_scope() -> PyrucastError {
    PyrucastError::Message(
        "a Handle can only be serialized inside an archive: its value is an \
         address, which means nothing outside this process. Use \
         archive::save / archive::load."
            .into(),
    )
}

// ─── Writing ────────────────────────────────────────────────────────────────

/// Run `f` with a write scope installed, and hand back the records it produced,
/// in dependency order.
pub(crate) fn with_write<R>(f: impl FnOnce() -> Result<R>) -> Result<(R, Vec<Record>)> {
    WRITING.with(|w| *w.borrow_mut() = Some(WriteScope::default()));
    let out = f();
    let scope = WRITING.with(|w| w.borrow_mut().take());
    let value = out?;
    let mut scope = scope.expect("the write scope is installed just above");
    // In deposit order: every dependency precedes what references it, so
    // reading is a plain forward loop.
    let mut records = Vec::with_capacity(scope.order.len());
    for id in std::mem::take(&mut scope.order) {
        match std::mem::replace(&mut scope.slots[id as usize], Slot::Reserved("taken")) {
            Slot::Written(r) => records.push(r),
            // Unreachable: an identifier only enters `order` once deposited.
            Slot::Reserved(tag) => unreachable!("record for {tag} was never deposited"),
        }
    }
    Ok((value, records))
}

/// The file-local identifier of `h`, writing its object out the first time.
///
/// Re-entrant: serializing the object walks its own handles through this same
/// function.
pub(crate) fn intern_handle<T: Archivable>(h: &Handle<T>) -> Result<u32> {
    let key = h.id();

    // Already interned? Then either it is complete, or we are inside its own
    // walk — which is a cycle.
    let known = WRITING.with(|w| -> Result<Option<u32>> {
        let mut b = w.borrow_mut();
        let scope = b.as_mut().ok_or_else(no_scope)?;
        Ok(scope.seen.get(&key).copied())
    })?;
    if let Some(id) = known {
        let complete = WRITING.with(|w| {
            matches!(
                w.borrow().as_ref().map(|s| &s.slots[id as usize]),
                Some(Slot::Written(_))
            )
        });
        if !complete {
            return Err(PyrucastError::Message(format!(
                "the object graph has a cycle: {h} refers back to itself \
                 through its own contents. Only recomputable caches may point \
                 backwards, and caches are not written."
            )));
        }
        return Ok(id);
    }

    // Reserve, then walk. The borrow is released before serializing, because
    // serializing re-enters this function.
    let id = WRITING.with(|w| -> Result<u32> {
        let mut b = w.borrow_mut();
        let scope = b.as_mut().ok_or_else(no_scope)?;
        let id = u32::try_from(scope.slots.len()).map_err(|_| {
            PyrucastError::Message("more than 4 billion objects in one archive".into())
        })?;
        scope.seen.insert(key, id);
        scope.slots.push(Slot::Reserved(T::TAG));
        Ok(id)
    })?;

    let payload = h.read().to_bytes()?;

    WRITING.with(|w| {
        if let Some(scope) = w.borrow_mut().as_mut() {
            scope.slots[id as usize] = Slot::Written(Record {
                id,
                tag: T::TAG,
                payload,
            });
            scope.order.push(id);
        }
    });
    Ok(id)
}

// ─── Reading ────────────────────────────────────────────────────────────────

/// Run `f` with a read scope installed.
pub(crate) fn with_read<R>(f: impl FnOnce() -> Result<R>) -> Result<R> {
    READING.with(|r| *r.borrow_mut() = Some(ReadScope::default()));
    let out = f();
    READING.with(|r| *r.borrow_mut() = None);
    out
}

/// Record a decoded object under its file-local identifier.
pub(crate) fn publish(id: u32, tag: String, object: Box<dyn Any + Send + Sync>) -> Result<()> {
    READING.with(|r| {
        let mut b = r.borrow_mut();
        let scope = b.as_mut().ok_or_else(no_scope)?;
        scope.objects.insert(id, (tag, object));
        Ok(())
    })
}

/// The tag a published object was decoded under.
pub(crate) fn tag_of(id: u32) -> Result<String> {
    READING.with(|r| {
        let b = r.borrow();
        let scope = b.as_ref().ok_or_else(no_scope)?;
        scope
            .objects
            .get(&id)
            .map(|(tag, _)| tag.clone())
            .ok_or_else(|| {
                PyrucastError::Message(format!("archive: root points at unknown object #{id}"))
            })
    })
}

/// The object behind a file-local identifier, as a fresh reference to it.
pub(crate) fn resolve<T: Archivable>(id: u32) -> Result<Handle<T>> {
    READING.with(|r| {
        let b = r.borrow();
        let scope = b.as_ref().ok_or_else(no_scope)?;
        let (_, any) = scope.objects.get(&id).ok_or_else(|| {
            PyrucastError::Message(format!(
                "archive: object #{id} is referenced before it is defined — \
                 the file is damaged or was written by another tool"
            ))
        })?;
        any.downcast_ref::<Handle<T>>().cloned().ok_or_else(|| {
            PyrucastError::Message(format!(
                "archive: object #{id} is not a {} — the file is damaged",
                T::TAG
            ))
        })
    })
}

/// Lend a published object to `f`, for the pass that turns roots into
/// [`Root`](super::Root) values.
pub(crate) fn with_object<R>(id: u32, f: impl FnOnce(&(dyn Any + Send + Sync)) -> R) -> Result<R> {
    READING.with(|r| {
        let b = r.borrow();
        let scope = b.as_ref().ok_or_else(no_scope)?;
        let (_, any) = scope.objects.get(&id).ok_or_else(|| {
            PyrucastError::Message(format!("archive: root points at unknown object #{id}"))
        })?;
        Ok(f(any.as_ref()))
    })
}

// ─── The two impls that make a handle a number ──────────────────────────────

impl<T: Archivable> serde::Serialize for Handle<T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::Error;
        let id = intern_handle(self).map_err(S::Error::custom)?;
        s.serialize_u32(id)
    }
}

impl<'de, T: Archivable> serde::Deserialize<'de> for Handle<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        use serde::de::Error;
        let id = u32::deserialize(d)?;
        resolve::<T>(id).map_err(D::Error::custom)
    }
}
