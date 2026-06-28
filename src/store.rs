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
//!   automatically on the next [`read`] / [`write`](fn@write). The binary format
//!   used is that of the [`crate::persist::Persist`] trait (portable
//!   between Linux and Windows).
//! - [`compact`] trims trailing free slots and shrinks memory.
//!
//! # Swap safety with respect to `Drop`
//!
//! Many pyrucast objects carry side effects in their `Drop` (for instance,
//! `SubMesh` decrements per-node refcounts inside the `Coords`).
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
//! Locking is **per object**, not per type. Each slot carries its own
//! [`parking_lot::RwLock`]; the store-level mutex is only held for the
//! instant it takes to resolve a handle into its slot (and on
//! insert/recycle). [`read`] returns a shared guard (many concurrent
//! readers), [`write`](fn@write) an exclusive one. Guards are *owned* (`'static`):
//! they can be returned from functions and stored in structs, so
//! operators read the data **in place** instead of copying it out.
//!
//! One usage rule applies: **do not acquire a second guard on an object
//! while holding a write guard on that same object** (and do not `write`
//! an object while holding any guard on it) — the slot lock is not
//! reentrant. Distinct objects, even of the same type, are fully
//! independent.
//!
//! # Example
//!
//! ```
//! use pyrucast::store::{insert, read, write};
//!
//! #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
//! struct Thing(Vec<f64>);
//!
//! let h = insert(Thing(vec![1.0, 2.0]));
//! assert_eq!(read(&h).unwrap().0, vec![1.0, 2.0]);
//! write(&h).unwrap().0.push(3.0);
//! assert_eq!(read(&h).unwrap().0.len(), 3);
//! ```

use crate::error::{PyrucastError, Result};
use crate::persist::Persist;
use parking_lot::lock_api::{
    ArcRwLockReadGuard, ArcRwLockUpgradableReadGuard, ArcRwLockWriteGuard,
};
use parking_lot::{Mutex, RawRwLock, RwLock};
use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

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
    *swap_dir_cell().lock() = Some(path.as_ref().to_path_buf());
}

/// Return the effective swap directory, creating it if necessary.
pub fn swap_dir() -> Result<PathBuf> {
    let mut guard = swap_dir_cell().lock();
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

fn store_for<T: Any + Send + Sync>() -> Arc<Mutex<StoreInner<T>>> {
    let mut reg = registry().lock();
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
    /// Post-extraction placeholder: the value left the cell (slot
    /// recycled). Never observed through a live handle.
    Free,
}

/// Shared per-slot cell: the object's own lock plus the handle refcount.
/// Lives in an `Arc` so that handles and guards keep it reachable without
/// going through the store mutex.
struct SlotCell<T> {
    lock: Arc<RwLock<SlotState<T>>>,
    refcount: AtomicU32,
}

struct Slot<T> {
    cell: Arc<SlotCell<T>>,
    gen: u32,
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

    fn insert(&mut self, cell: Arc<SlotCell<T>>) -> (u32, u32) {
        if let Some(idx) = self.free.pop() {
            let s = &mut self.slots[idx as usize];
            s.cell = cell;
            s.gen = s.gen.wrapping_add(1);
            (idx, s.gen)
        } else {
            let idx = self.slots.len() as u32;
            self.slots.push(Slot { cell, gen: 1 });
            (idx, 1)
        }
    }

    /// Resolve `(idx, gen)` into the slot's cell. A slot is live iff the
    /// generation matches and at least one handle still points to it.
    fn resolve(&self, idx: u32, gen: u32) -> Result<Arc<SlotCell<T>>> {
        let s = self
            .slots
            .get(idx as usize)
            .ok_or(PyrucastError::StaleHandle)?;
        if s.gen != gen || s.cell.refcount.load(Ordering::Acquire) == 0 {
            return Err(PyrucastError::StaleHandle);
        }
        Ok(s.cell.clone())
    }

    fn compact(&mut self) {
        while self
            .slots
            .last()
            .is_some_and(|s| s.cell.refcount.load(Ordering::Acquire) == 0)
        {
            self.slots.pop();
        }
        self.free.retain(|&i| (i as usize) < self.slots.len());
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
/// through disk without breaking refcounts (the serialized handle's
/// count is carried by the on-disk object and reclaimed on reload).
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Handle<T: Persist + Any + Send + Sync> {
    idx: u32,
    gen: u32,
    /// Cached pointer to the slot cell, resolved lazily (a freshly
    /// deserialized handle starts empty). Skipped by serde.
    #[serde(skip)]
    cell: OnceLock<Arc<SlotCell<T>>>,
}

impl<T: Persist + Any + Send + Sync> Handle<T> {
    /// Internal index (useful for debugging and display).
    pub fn index(&self) -> u32 {
        self.idx
    }
    /// Current generation of the pointed-to slot.
    pub fn generation(&self) -> u32 {
        self.gen
    }

    /// Whether two handles designate the **same store slot** — same index
    /// *and* same generation (a recycled slot has a fresh generation, so
    /// stale handles never compare equal to a live one).
    ///
    /// This is handle *identity*, not value equality: it is the basis of
    /// the aggregates' union (`a | b` skips a sub whose handle is already
    /// present). Deliberately not a `PartialEq` impl, to keep handle
    /// comparison an explicit, intentional act.
    pub fn same_slot(&self, other: &Self) -> bool {
        self.idx == other.idx && self.gen == other.gen
    }

    /// Resolve (and cache) the slot cell. The store mutex is held only
    /// for the lookup itself.
    fn resolve(&self) -> Result<Arc<SlotCell<T>>> {
        if let Some(c) = self.cell.get() {
            return Ok(c.clone());
        }
        let store = store_for::<T>();
        let cell = store.lock().resolve(self.idx, self.gen)?;
        let _ = self.cell.set(cell.clone());
        Ok(cell)
    }
}

impl<T: Persist + Any + Send + Sync> Clone for Handle<T> {
    fn clone(&self) -> Self {
        // While `&self` exists the refcount is ≥ 1, so the slot cannot be
        // recycled under us: a plain atomic increment suffices.
        let cache = OnceLock::new();
        if let Ok(cell) = self.resolve() {
            cell.refcount.fetch_add(1, Ordering::AcqRel);
            let _ = cache.set(cell);
        }
        Self {
            idx: self.idx,
            gen: self.gen,
            cell: cache,
        }
    }
}

impl<T: Persist + Any + Send + Sync> Drop for Handle<T> {
    fn drop(&mut self) {
        let Ok(cell) = self.resolve() else { return };
        if cell.refcount.fetch_sub(1, Ordering::AcqRel) == 1 {
            release_slot::<T>(self.idx, self.gen);
        }
    }
}

impl<T: Persist + Any + Send + Sync> fmt::Debug for Handle<T> {
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

impl<T: Persist + Any + Send + Sync> fmt::Display for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let short = std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap_or("?");
        write!(f, "<{} #{}>", short, self.idx)
    }
}

/// Recycle a slot whose refcount reached 0: mark it free under the store
/// mutex, extract the value, then run its `Drop` **outside every lock**
/// (Drop side effects touch other stores).
fn release_slot<T: Persist + Any + Send + Sync>(idx: u32, gen: u32) {
    let store = store_for::<T>();
    let mut inner = store.lock();
    let Some(slot) = inner.slots.get_mut(idx as usize) else {
        return;
    };
    if slot.gen != gen || slot.cell.refcount.load(Ordering::Acquire) != 0 {
        return;
    }
    // No handle ⇒ no guard (guards carry a handle), so this write lock is
    // uncontended.
    let old = {
        let mut w = slot.cell.lock.write();
        std::mem::replace(&mut *w, SlotState::Free)
    };
    inner.free.push(idx);
    drop(inner);
    match old {
        SlotState::Resident(v) => drop(v),
        SlotState::OnDisk(path) => {
            // Reload to run Drop properly. On I/O or deserialization
            // failure, the slot's Drop side effects are lost (documented
            // limitation).
            if let Ok(bytes) = std::fs::read(&path) {
                let _ = std::fs::remove_file(&path);
                if let Ok(v) = T::from_bytes(&bytes) {
                    drop(v);
                }
            }
        }
        SlotState::Free => {}
    }
}

// ─── Guards ─────────────────────────────────────────────────────────────────

/// Shared (read) access to a stored object, returned by [`read`].
///
/// Owned (`'static`): can be returned from functions and stored in
/// structs. Dereferences to `&T`; the slot lock is released — and the
/// keepalive refcount dropped — when the guard goes out of scope.
pub struct ReadGuard<T: Persist + Any + Send + Sync> {
    guard: ArcRwLockReadGuard<RawRwLock, SlotState<T>>,
    /// Keeps the slot alive while the guard lives. Declared after
    /// `guard`: the lock is released before the refcount drops.
    _keepalive: Handle<T>,
}

impl<T: Persist + Any + Send + Sync> std::ops::Deref for ReadGuard<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match &*self.guard {
            SlotState::Resident(v) => v,
            // The state cannot change while we hold the slot lock, and
            // acquisition guaranteed Resident.
            _ => unreachable!("ReadGuard over a non-resident slot"),
        }
    }
}

impl<T: Persist + Any + Send + Sync + fmt::Debug> fmt::Debug for ReadGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

/// Exclusive (write) access to a stored object, returned by [`write`](fn@write).
/// Same ownership properties as [`ReadGuard`].
pub struct WriteGuard<T: Persist + Any + Send + Sync> {
    guard: ArcRwLockWriteGuard<RawRwLock, SlotState<T>>,
    _keepalive: Handle<T>,
}

impl<T: Persist + Any + Send + Sync> std::ops::Deref for WriteGuard<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match &*self.guard {
            SlotState::Resident(v) => v,
            _ => unreachable!("WriteGuard over a non-resident slot"),
        }
    }
}

impl<T: Persist + Any + Send + Sync> std::ops::DerefMut for WriteGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        match &mut *self.guard {
            SlotState::Resident(v) => v,
            _ => unreachable!("WriteGuard over a non-resident slot"),
        }
    }
}

impl<T: Persist + Any + Send + Sync + fmt::Debug> fmt::Debug for WriteGuard<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

// ─── Public store API ───────────────────────────────────────────────────────

/// Insert a value into the global store for its type and return a handle.
pub fn insert<T: Persist + Any + Send + Sync>(value: T) -> Handle<T> {
    let cell = Arc::new(SlotCell {
        lock: Arc::new(RwLock::new(SlotState::Resident(value))),
        refcount: AtomicU32::new(1),
    });
    let store = store_for::<T>();
    let (idx, gen) = store.lock().insert(cell.clone());
    let cache = OnceLock::new();
    let _ = cache.set(cell);
    Handle {
        idx,
        gen,
        cell: cache,
    }
}

/// Reload an `OnDisk` state in place. No-op when already resident.
fn reload<T: Persist>(state: &mut SlotState<T>) -> Result<()> {
    if let SlotState::OnDisk(path) = state {
        let bytes = std::fs::read(&path)?;
        let value = T::from_bytes(&bytes)?;
        let path = path.clone();
        *state = SlotState::Resident(value);
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}

/// Shared read access: many readers may hold guards on the same object
/// concurrently. Reloads from disk if the slot was evicted.
pub fn read<T: Persist + Any + Send + Sync>(h: &Handle<T>) -> Result<ReadGuard<T>> {
    let cell = h.resolve()?;
    // Fast path: a shared lock suffices when the value is resident.
    let g = cell.lock.read_arc();
    let guard = if matches!(&*g, SlotState::OnDisk(_)) {
        drop(g);
        // Slow path: reload under an upgradable lock, then downgrade.
        let up = cell.lock.upgradable_read_arc();
        if matches!(&*up, SlotState::OnDisk(_)) {
            let mut w = ArcRwLockUpgradableReadGuard::upgrade(up);
            reload(&mut *w)?;
            ArcRwLockWriteGuard::downgrade(w)
        } else {
            ArcRwLockUpgradableReadGuard::downgrade(up)
        }
    } else {
        g
    };
    Ok(ReadGuard {
        guard,
        _keepalive: h.clone(),
    })
}

/// Exclusive write access. Reloads from disk if necessary.
pub fn write<T: Persist + Any + Send + Sync>(h: &Handle<T>) -> Result<WriteGuard<T>> {
    let cell = h.resolve()?;
    let mut g = cell.lock.write_arc();
    reload(&mut *g)?;
    Ok(WriteGuard {
        guard: g,
        _keepalive: h.clone(),
    })
}

/// Evict the slot to disk (freeing its RAM). The slot stays valid; the
/// next [`read`] / [`write`](fn@write) will reload.
///
/// **Important**: the evicted value's `Drop` does **not** run (the object
/// is still logically alive). It will run on the final refcount
/// decrement, reloading from disk first if necessary.
pub fn swap_out<T: Persist + Any + Send + Sync>(h: &Handle<T>) -> Result<()> {
    let cell = h.resolve()?;
    let mut w = cell.lock.write();
    let bytes = match &*w {
        SlotState::Resident(v) => v.to_bytes()?,
        SlotState::OnDisk(_) => return Ok(()),
        SlotState::Free => return Err(PyrucastError::StaleHandle),
    };
    let dir = swap_dir()?;
    let type_id = TypeId::of::<T>();
    let path = dir.join(format!("slot-{:?}-{}-{}.bin", type_id, h.idx, h.gen));
    std::fs::write(&path, bytes)?;
    let old = std::mem::replace(&mut *w, SlotState::OnDisk(path));
    // Object stays logically alive: bypass Drop so we do not trigger side
    // effects (refcounts, files, …). Drop will run on the final
    // refcount decrement.
    std::mem::forget(old);
    Ok(())
}

/// Compact the store of type T: trim trailing free slots and shrink
/// the associated memory.
pub fn compact<T: Any + Send + Sync>() {
    let store = store_for::<T>();
    store.lock().compact();
}

/// Capacity (number of slots) of the store for type T.
pub fn capacity<T: Any + Send + Sync>() -> usize {
    let store = store_for::<T>();
    let inner = store.lock();
    inner.slots.len()
}

/// Number of live (non-free) slots for type T.
pub fn live_count<T: Any + Send + Sync>() -> usize {
    let store = store_for::<T>();
    let inner = store.lock();
    inner
        .slots
        .iter()
        .filter(|s| s.cell.refcount.load(Ordering::Acquire) > 0)
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
        let h = insert(PInsertGet(1.5));
        assert_eq!(read(&h).unwrap().0, 1.5);
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
        assert_eq!(read(&h2).unwrap().0, 42);
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
            cell: OnceLock::new(),
        };
        let err = read(&stale).unwrap_err();
        assert!(matches!(err, PyrucastError::StaleHandle));
        std::mem::forget(stale);
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PMut(Vec<u32>);

    #[test]
    fn write_modifies_in_place() {
        let h = insert(PMut(vec![1, 2]));
        write(&h).unwrap().0.push(3);
        assert_eq!(read(&h).unwrap().0, vec![1, 2, 3]);
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct PSwap(Vec<f64>);

    #[test]
    fn swap_out_then_access_reloads() {
        let h = insert(PSwap(vec![10.0, 20.0, 30.0]));
        swap_out(&h).unwrap();
        assert_eq!(read(&h).unwrap().0, vec![10.0, 20.0, 30.0]);
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

    // ─── Per-slot locking ───────────────────────────────────────────────

    #[derive(Serialize, Deserialize, Debug)]
    struct PNest(u32);

    /// Guards on two distinct objects of the same type may coexist —
    /// deadlocked with the old per-type mutex.
    #[test]
    fn guards_on_distinct_objects_same_type() {
        let a = insert(PNest(1));
        let b = insert(PNest(2));
        let ga = read(&a).unwrap();
        let gb = read(&b).unwrap();
        assert_eq!(ga.0 + gb.0, 3);
        // Write on a third object while holding the two reads.
        let c = insert(PNest(0));
        write(&c).unwrap().0 = ga.0;
        assert_eq!(read(&c).unwrap().0, 1);
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PCloneUnderGuard(u8);

    /// Cloning a handle while a guard on the same object is held —
    /// deadlocked with the old per-type mutex (refcount under the mutex).
    #[test]
    fn clone_handle_while_guard_held() {
        let a = insert(PCloneUnderGuard(7));
        let g = read(&a).unwrap();
        let b = a.clone();
        assert_eq!(g.0, 7);
        drop(g);
        assert_eq!(read(&b).unwrap().0, 7);
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PKeepAlive(u8);

    /// A guard keeps the slot alive past the drop of the last handle.
    #[test]
    fn guard_outlives_last_handle() {
        let a = insert(PKeepAlive(9));
        let g = read(&a).unwrap();
        drop(a);
        assert_eq!(g.0, 9);
        assert_eq!(live_count::<PKeepAlive>(), 1);
        drop(g); // slot released here
        assert_eq!(live_count::<PKeepAlive>(), 0);
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PPar(Vec<f64>);

    /// Several threads may read the same object concurrently.
    #[test]
    fn concurrent_readers() {
        let h = insert(PPar(vec![1.0; 1000]));
        std::thread::scope(|s| {
            for _ in 0..4 {
                let h = h.clone();
                s.spawn(move || {
                    let g = read(&h).unwrap();
                    assert_eq!(g.0.len(), 1000);
                });
            }
        });
    }

    #[derive(Serialize, Deserialize, Debug)]
    struct PSwapGuard(Vec<u8>);

    /// Reload-on-read still works when the guard is kept around.
    #[test]
    fn guard_after_swap_reload() {
        let h = insert(PSwapGuard(vec![1, 2, 3]));
        swap_out(&h).unwrap();
        let g = read(&h).unwrap();
        assert_eq!(g.0, vec![1, 2, 3]);
        // A second reader joins while the first guard is alive.
        let g2 = read(&h).unwrap();
        assert_eq!(g2.0, g.0);
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

        // Resident path → swap_out → read → final drop
        let h = insert(PSwapDrop);
        swap_out(&h).unwrap();
        assert_eq!(
            SWAP_DROP_COUNT.load(Ordering::SeqCst),
            0,
            "swap_out must NOT run Drop"
        );
        drop(read(&h).unwrap());
        assert_eq!(
            SWAP_DROP_COUNT.load(Ordering::SeqCst),
            0,
            "reload must NOT run Drop"
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
