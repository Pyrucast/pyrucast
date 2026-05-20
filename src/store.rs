//! Global typed store — pyrucast's "object pile", cast3m-style.
//!
//! Every object lives in a **process-global** store, internally indexed by
//! [`std::any::TypeId`]: one `Store<T>` per type `T`. The public API is a
//! set of module-level functions (no Session to pass around).
//!
//! # Guarantees
//!
//! - [`Handle<T>`] is **generational**: a recycled slot invalidates every
//!   prior handle (access returns [`PyrucastError::StaleHandle`]).
//! - [`Handle<T>`] is **refcounted**: `Clone` increments, `Drop`
//!   decrements, and the slot is recycled automatically when it reaches 0.
//! - A slot may be evicted to disk via [`swap_out`] and is reloaded
//!   automatically on the next [`with`] / [`with_mut`]. The binary format
//!   used is that of the [`crate::persist::Persist`] trait (portable
//!   between Linux and Windows).
//! - [`compact`] trims trailing free slots and shrinks memory.
//!
//! # Swap safety with respect to `Drop`
//!
//! Many pyrucast objects carry side effects in their `Drop` (for instance,
//! `SubMesh` decrements per-node refcounts inside the `Configuration`).
//! The swap path is designed **not** to trigger `Drop` on eviction: the
//! object is still logically alive, just relocated.
//!  - [`swap_out`] serializes then swaps the Resident state for OnDisk,
//!    "forgetting" the old value (via [`std::mem::forget`]) — Drop does
//!    not run.
//!  - When the refcount reaches 0 on an OnDisk slot, the value is reloaded
//!    from disk before being dropped, so Drop side effects fire exactly
//!    once over the object's lifetime.
//!
//! # Concurrency
//!
//! The per-type inner store is protected by a [`std::sync::Mutex`]. One
//! usage rule applies: **do not perform any operation on the same type
//! `T` from within a closure passed to [`with`] / [`with_mut`]**
//! (reentrancy on the same mutex → deadlock). Operations on different
//! types are independent.
//!
//! # Example
//!
//! ```
//! use pyrucast::store::{insert, with, with_mut};
//!
//! #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
//! struct Thing(Vec<f64>);
//!
//! let h = insert(Thing(vec![1.0, 2.0]));
//! with(&h, |t| assert_eq!(t.0, vec![1.0, 2.0])).unwrap();
//! with_mut(&h, |t| t.0.push(3.0)).unwrap();
//! with(&h, |t| assert_eq!(t.0.len(), 3)).unwrap();
//! ```

use crate::error::{PyrucastError, Result};
use crate::persist::Persist;
use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

// ─── Global swap configuration ──────────────────────────────────────────────

static SWAP_DIR: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

fn swap_dir_cell() -> &'static Mutex<Option<PathBuf>> {
    SWAP_DIR.get_or_init(|| Mutex::new(None))
}

/// Set the directory in which evicted slots are serialized.
///
/// If never called, a per-process subdirectory of [`std::env::temp_dir`]
/// is used.
pub fn set_swap_dir(path: impl AsRef<Path>) {
    *swap_dir_cell().lock().expect("poisoned mutex") = Some(path.as_ref().to_path_buf());
}

/// Return the effective swap directory, creating it if necessary.
pub fn swap_dir() -> Result<PathBuf> {
    let mut guard = swap_dir_cell().lock().expect("poisoned mutex");
    if guard.is_none() {
        let p = std::env::temp_dir().join(format!("pyrucast-swap-{}", std::process::id()));
        std::fs::create_dir_all(&p)?;
        *guard = Some(p);
    }
    Ok(guard.as_ref().unwrap().clone())
}

// ─── Registry of per-type stores (one per TypeId) ───────────────────────────

type AnyStore = Box<dyn Any + Send>;

static REGISTRY: OnceLock<Mutex<HashMap<TypeId, AnyStore>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<TypeId, AnyStore>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn store_for<T: Any + Send>() -> Arc<Mutex<StoreInner<T>>> {
    let mut reg = registry().lock().expect("poisoned mutex");
    let entry = reg
        .entry(TypeId::of::<T>())
        .or_insert_with(|| Box::new(Arc::new(Mutex::new(StoreInner::<T>::new()))));
    entry
        .downcast_ref::<Arc<Mutex<StoreInner<T>>>>()
        .expect("TypeId collision in registry")
        .clone()
}

// ─── Internal slot ──────────────────────────────────────────────────────────

enum SlotState<T> {
    Resident(T),
    OnDisk(PathBuf),
    Free,
}

struct Slot<T> {
    state: SlotState<T>,
    gen: u32,
    refcount: u32,
}

struct StoreInner<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> StoreInner<T> {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    fn insert(&mut self, value: T) -> (u32, u32) {
        if let Some(idx) = self.free.pop() {
            let s = &mut self.slots[idx as usize];
            s.state = SlotState::Resident(value);
            s.gen = s.gen.wrapping_add(1);
            s.refcount = 1;
            (idx, s.gen)
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot {
                state: SlotState::Resident(value),
                gen: 1,
                refcount: 1,
            });
            (idx, 1)
        }
    }

    fn validate(&self, idx: u32, gen: u32) -> Result<()> {
        let s = self.slots.get(idx as usize).ok_or(PyrucastError::StaleHandle)?;
        if s.gen != gen || matches!(s.state, SlotState::Free) {
            return Err(PyrucastError::StaleHandle);
        }
        Ok(())
    }

    fn incref(&mut self, idx: u32, gen: u32) -> Result<()> {
        self.validate(idx, gen)?;
        self.slots[idx as usize].refcount = self.slots[idx as usize].refcount.saturating_add(1);
        Ok(())
    }

    fn compact(&mut self) {
        while matches!(self.slots.last().map(|s| &s.state), Some(SlotState::Free)) {
            self.slots.pop();
        }
        self.free.retain(|&i| (i as usize) < self.slots.len());
    }
}

impl<T: Persist> StoreInner<T> {
    /// Decrement the refcount; if it reaches 0, return the value to be
    /// dropped **outside the lock** (avoids any deadlock should `Drop`
    /// touch a store of the same type).
    ///
    /// If the state was `OnDisk`, the value is reloaded from disk before
    /// being returned for Drop — so the object's Drop side effects fire
    /// exactly once.
    fn decref(&mut self, idx: u32, gen: u32) -> Option<T> {
        if self.validate(idx, gen).is_err() {
            return None;
        }
        let s = &mut self.slots[idx as usize];
        s.refcount = s.refcount.saturating_sub(1);
        if s.refcount == 0 {
            let old = std::mem::replace(&mut s.state, SlotState::Free);
            self.free.push(idx);
            return match old {
                SlotState::Resident(v) => Some(v),
                SlotState::OnDisk(path) => {
                    // Reload to run Drop properly. On I/O or
                    // deserialization failure, the slot's Drop side
                    // effects are lost (documented limitation).
                    let bytes = std::fs::read(&path).ok()?;
                    let _ = std::fs::remove_file(path);
                    T::from_bytes(&bytes).ok()
                }
                SlotState::Free => None,
            };
        }
        None
    }
}

// ─── Public handle ──────────────────────────────────────────────────────────

/// Refcounted and generational reference to an object in the global store.
///
/// - `Clone` increments the refcount automatically.
/// - `Drop` decrements; when it reaches 0, the slot is recycled.
/// - `Debug` shows the structural view (idx + generation + type).
/// - `Display` shows the short cast3m-style view (`<ShortName #idx>`).
///
/// `Handle<T>` is serializable (idx + generation); combined with the
/// Drop-safe swap, this lets objects containing handles round-trip
/// through disk without breaking refcounts.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Handle<T: Persist + Any + Send> {
    idx: u32,
    gen: u32,
    #[serde(skip)]
    _t: PhantomData<fn() -> T>,
}

impl<T: Persist + Any + Send> Handle<T> {
    /// Internal index (useful for debugging and display).
    pub fn index(&self) -> u32 {
        self.idx
    }
    /// Current generation of the pointed-to slot.
    pub fn generation(&self) -> u32 {
        self.gen
    }
}

impl<T: Persist + Any + Send> Clone for Handle<T> {
    fn clone(&self) -> Self {
        let _ = store_for::<T>()
            .lock()
            .expect("poisoned mutex")
            .incref(self.idx, self.gen);
        Self {
            idx: self.idx,
            gen: self.gen,
            _t: PhantomData,
        }
    }
}

impl<T: Persist + Any + Send> Drop for Handle<T> {
    fn drop(&mut self) {
        let to_drop = store_for::<T>()
            .lock()
            .expect("poisoned mutex")
            .decref(self.idx, self.gen);
        drop(to_drop);
    }
}

impl<T: Persist + Any + Send> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Handle<{}>{{ idx: {}, gen: {} }}",
            std::any::type_name::<T>(),
            self.idx,
            self.gen
        )
    }
}

impl<T: Persist + Any + Send> fmt::Display for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let short = std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap_or("?");
        write!(f, "<{} #{}>", short, self.idx)
    }
}

// ─── Public store API ───────────────────────────────────────────────────────

/// Insert a value into the global store for its type and return a handle.
pub fn insert<T: Persist + Any + Send>(value: T) -> Handle<T> {
    let store = store_for::<T>();
    let (idx, gen) = store.lock().expect("poisoned mutex").insert(value);
    Handle {
        idx,
        gen,
        _t: PhantomData,
    }
}

/// Read access. Reloads from disk if the slot was evicted.
pub fn with<T: Persist + Any + Send, R>(h: &Handle<T>, f: impl FnOnce(&T) -> R) -> Result<R> {
    let store = store_for::<T>();
    let mut inner = store.lock().expect("poisoned mutex");
    ensure_resident::<T>(&mut inner, h.idx, h.gen)?;
    let s = &inner.slots[h.idx as usize];
    match &s.state {
        SlotState::Resident(v) => Ok(f(v)),
        _ => unreachable!("ensure_resident guarantees Resident state"),
    }
}

/// Write access. Reloads from disk if necessary.
pub fn with_mut<T: Persist + Any + Send, R>(
    h: &Handle<T>,
    f: impl FnOnce(&mut T) -> R,
) -> Result<R> {
    let store = store_for::<T>();
    let mut inner = store.lock().expect("poisoned mutex");
    ensure_resident::<T>(&mut inner, h.idx, h.gen)?;
    let s = &mut inner.slots[h.idx as usize];
    match &mut s.state {
        SlotState::Resident(v) => Ok(f(v)),
        _ => unreachable!("ensure_resident guarantees Resident state"),
    }
}

fn ensure_resident<T: Persist + Any + Send>(
    inner: &mut StoreInner<T>,
    idx: u32,
    gen: u32,
) -> Result<()> {
    inner.validate(idx, gen)?;
    let s = &mut inner.slots[idx as usize];
    if let SlotState::OnDisk(path) = &s.state {
        let bytes = std::fs::read(path)?;
        let value = T::from_bytes(&bytes)?;
        let path = path.clone();
        s.state = SlotState::Resident(value);
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

/// Evict the slot to disk (freeing its RAM). The slot stays valid; the
/// next [`with`] / [`with_mut`] will reload.
///
/// **Important**: the evicted value's `Drop` does **not** run (the object
/// is still logically alive). It will run on the final refcount
/// decrement, reloading from disk first if necessary.
pub fn swap_out<T: Persist + Any + Send>(h: &Handle<T>) -> Result<()> {
    let store = store_for::<T>();
    let mut inner = store.lock().expect("poisoned mutex");
    inner.validate(h.idx, h.gen)?;
    let s = &mut inner.slots[h.idx as usize];
    let bytes = match &s.state {
        SlotState::Resident(v) => v.to_bytes()?,
        SlotState::OnDisk(_) => return Ok(()),
        SlotState::Free => return Err(PyrucastError::StaleHandle),
    };
    let dir = swap_dir()?;
    let type_id = TypeId::of::<T>();
    let path = dir.join(format!("slot-{:?}-{}-{}.bin", type_id, h.idx, h.gen));
    std::fs::write(&path, bytes)?;
    let old = std::mem::replace(&mut s.state, SlotState::OnDisk(path));
    // Object stays logically alive: bypass Drop so we do not trigger side
    // effects (refcounts, files, …). Drop will run on the final
    // refcount decrement.
    match old {
        SlotState::Resident(v) => std::mem::forget(v),
        SlotState::OnDisk(_) | SlotState::Free => {}
    }
    Ok(())
}

/// Compact the store of type T: trim trailing free slots and shrink
/// the associated memory.
pub fn compact<T: Any + Send>() {
    let store = store_for::<T>();
    store.lock().expect("poisoned mutex").compact();
}

/// Capacity (number of slots) of the store for type T.
pub fn capacity<T: Any + Send>() -> usize {
    let store = store_for::<T>();
    let inner = store.lock().expect("poisoned mutex");
    inner.slots.len()
}

/// Number of live (non-free) slots for type T.
pub fn live_count<T: Any + Send>() -> usize {
    let store = store_for::<T>();
    let inner = store.lock().expect("poisoned mutex");
    inner
        .slots
        .iter()
        .filter(|s| !matches!(s.state, SlotState::Free))
        .count()
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    // Each test uses a unique newtype to isolate its global store.

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct PInsertGet(f64);

    #[test]
    fn insert_then_read() {
        let h = insert(PInsertGet(3.14));
        with(&h, |v| assert_eq!(v.0, 3.14)).unwrap();
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PClone(i32);

    #[test]
    fn clone_shares_slot() {
        let h1 = insert(PClone(42));
        let h2 = h1.clone();
        assert_eq!(h1.index(), h2.index());
        assert_eq!(live_count::<PClone>(), 1);
        drop(h1);
        with(&h2, |v| assert_eq!(v.0, 42)).unwrap();
        assert_eq!(live_count::<PClone>(), 1);
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PRecycle(String);

    #[test]
    fn last_drop_recycles_slot() {
        let h = insert(PRecycle("a".into()));
        let idx = h.index();
        let gen_before = h.generation();
        drop(h);
        let h2 = insert(PRecycle("b".into()));
        assert_eq!(h2.index(), idx);
        assert_ne!(h2.generation(), gen_before);
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PStale(u8);

    #[test]
    fn stale_handle_after_recycle() {
        let h = insert(PStale(1));
        let idx = h.index();
        let obsolete_gen = h.generation();
        drop(h);
        let _h2 = insert(PStale(2));
        let stale: Handle<PStale> = Handle {
            idx,
            gen: obsolete_gen,
            _t: PhantomData,
        };
        let err = with(&stale, |_| ()).unwrap_err();
        assert!(matches!(err, PyrucastError::StaleHandle));
        std::mem::forget(stale);
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PMut(Vec<u32>);

    #[test]
    fn with_mut_modifies_in_place() {
        let h = insert(PMut(vec![1, 2]));
        with_mut(&h, |v| v.0.push(3)).unwrap();
        with(&h, |v| assert_eq!(v.0, vec![1, 2, 3])).unwrap();
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct PSwap(Vec<f64>);

    #[test]
    fn swap_out_then_access_reloads() {
        let h = insert(PSwap(vec![10.0, 20.0, 30.0]));
        swap_out(&h).unwrap();
        with(&h, |v| assert_eq!(v.0, vec![10.0, 20.0, 30.0])).unwrap();
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PCompact(u64);

    #[test]
    fn compact_trims_trailing_free_slots() {
        let a = insert(PCompact(1));
        let b = insert(PCompact(2));
        let c = insert(PCompact(3));
        assert_eq!(capacity::<PCompact>(), 3);
        drop(c);
        drop(b);
        drop(a);
        assert_eq!(live_count::<PCompact>(), 0);
        compact::<PCompact>();
        assert_eq!(capacity::<PCompact>(), 0);
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PDisplay;

    #[test]
    fn handle_display() {
        let h = insert(PDisplay);
        let dbg = format!("{:?}", h);
        let dsp = format!("{}", h);
        assert!(dbg.contains("PDisplay"));
        assert!(dbg.contains("idx:"));
        assert!(dsp.starts_with("<PDisplay #"));
    }

    // ─── Swap safety w.r.t. Drop ────────────────────────────────────────

    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Serialize, Deserialize)]
    struct PSwapDrop;
    static SWAP_DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
    impl Drop for PSwapDrop {
        fn drop(&mut self) {
            SWAP_DROP_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn swap_preserves_drop_of_objects() {
        SWAP_DROP_COUNT.store(0, Ordering::SeqCst);

        // Resident path → swap_out → with → final drop
        let h = insert(PSwapDrop);
        swap_out(&h).unwrap();
        assert_eq!(
            SWAP_DROP_COUNT.load(Ordering::SeqCst),
            0,
            "swap_out must NOT run Drop"
        );
        with(&h, |_| ()).unwrap();
        assert_eq!(
            SWAP_DROP_COUNT.load(Ordering::SeqCst),
            0,
            "ensure_resident must NOT run Drop"
        );
        drop(h);
        assert_eq!(
            SWAP_DROP_COUNT.load(Ordering::SeqCst),
            1,
            "final Drop from Resident must run exactly once"
        );

        // OnDisk path → final drop (no prior reload)
        let h2 = insert(PSwapDrop);
        swap_out(&h2).unwrap();
        assert_eq!(SWAP_DROP_COUNT.load(Ordering::SeqCst), 1);
        drop(h2);
        assert_eq!(
            SWAP_DROP_COUNT.load(Ordering::SeqCst),
            2,
            "final Drop from OnDisk must reload and run exactly once"
        );
    }
}
