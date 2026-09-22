//! User-space swap: large allocations live in file-backed mappings.
//!
//! A machine without swap, and without the root access that adding a swapfile
//! needs, kills a process as soon as its anonymous memory exceeds the RAM.
//! Pages of a **shared file mapping** are not anonymous: under memory pressure
//! the kernel writes them back to their file and evicts them, then faults them
//! in again on access — which is what swap does, with no privilege at all.
//!
//! [`SpillAlloc`] is the process-wide allocator under the `spill` feature
//! (Linux only). Every allocation of at least `PYRUCAST_SPILL_MIN` bytes
//! (default 64 MiB) goes to its own unnamed temporary file created in
//! `PYRUCAST_SPILL_DIR` (`O_TMPFILE`: nothing to clean up, even after a crash);
//! everything smaller, and everything when the variable is unset, goes to the
//! system allocator. faer allocates its factors through the global allocator,
//! so a factorization spills without knowing it.
//!
//! What it costs: a written page of a file mapping is flushed to disk after
//! about thirty seconds even without pressure (`vm.dirty_expire_centisecs`, a
//! root-only setting), so spilling does I/O even when the RAM would have been
//! enough. Hence opt-in, per run.
//!
//! [`stats`] tells what was spilled: how many blocks, how large, how much is
//! mapped now and at most. With `PYRUCAST_SPILL_LOG` set as well, every mapping
//! and unmapping is also written to stderr as it happens — the list of the
//! blocks, one line each.
//!
//! The variables are read **once**, by the first allocation of the process —
//! set them before starting it (before `import pyrucast` in Python). A
//! directory that cannot be opened aborts the process with a message: the
//! user asked for spilling, and silently running without it would only move
//! the failure to the out-of-memory killer.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};

/// Spilling threshold in bytes. `0` until the environment has been read — so
/// that the very first allocation takes the slow path and reads it — then the
/// configured value, or `usize::MAX` when spilling is off.
static THRESHOLD: AtomicUsize = AtomicUsize::new(0);

/// Directory the spill files are created in, opened once.
static DIR: AtomicI32 = AtomicI32::new(-1);

/// Whether each mapping and unmapping is written to stderr.
static LOG: AtomicBool = AtomicBool::new(false);

// The counters behind [`stats`]. Touched on the spilling path only — the
// allocations below the threshold never see them.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

/// What the spilling allocator has done since the process started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpillStats {
    /// Allocations at least this large spill; `usize::MAX` when spilling is off.
    pub threshold: usize,
    /// Blocks mapped so far, freed ones included.
    pub count: usize,
    /// The largest block mapped, in bytes.
    pub largest: usize,
    /// Bytes mapped right now.
    pub live: usize,
    /// The most bytes ever mapped at once.
    pub peak: usize,
}

/// The allocator's counters, read now.
///
/// ```
/// let s = pyrucast::spill::stats();
/// // Nothing can be mapped without having been counted, nor exceed its peak.
/// assert!(s.live <= s.peak);
/// assert!(s.count > 0 || s.peak == 0);
/// ```
pub fn stats() -> SpillStats {
    SpillStats {
        threshold: THRESHOLD.load(Ordering::Acquire),
        count: COUNT.load(Ordering::Relaxed),
        largest: LARGEST.load(Ordering::Relaxed),
        live: LIVE.load(Ordering::Relaxed),
        peak: PEAK.load(Ordering::Relaxed),
    }
}

/// A mapping is page-aligned; a layout asking for more cannot be one.
const PAGE: usize = 4096;

const DEFAULT_THRESHOLD: usize = 64 << 20;

/// The global allocator of the `spill` feature — see the [module](self).
pub struct SpillAlloc;

/// Read the environment and publish the threshold. Idempotent: two threads
/// racing here read the same environment and store the same values.
#[cold]
fn configure() -> usize {
    // SAFETY: `getenv` returns null or a NUL-terminated string that stays valid
    // as long as nobody calls `setenv` concurrently — the documented contract.
    let dir = unsafe { libc::getenv(c"PYRUCAST_SPILL_DIR".as_ptr()) };
    if dir.is_null() {
        THRESHOLD.store(usize::MAX, Ordering::Relaxed);
        return usize::MAX;
    }
    // SAFETY: `dir` is a valid C string (above).
    let fd = unsafe { libc::open(dir, libc::O_DIRECTORY | libc::O_RDONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        die(b"pyrucast: PYRUCAST_SPILL_DIR cannot be opened as a directory\n");
    }
    // SAFETY: as for `dir`.
    let min = unsafe { libc::getenv(c"PYRUCAST_SPILL_MIN".as_ptr()) };
    let threshold = if min.is_null() {
        DEFAULT_THRESHOLD
    } else {
        // SAFETY: a valid C string (above).
        parse_bytes(unsafe { std::ffi::CStr::from_ptr(min) }.to_bytes())
    };
    // SAFETY: as for `dir`.
    LOG.store(
        !unsafe { libc::getenv(c"PYRUCAST_SPILL_LOG".as_ptr()) }.is_null(),
        Ordering::Relaxed,
    );
    if DIR
        .compare_exchange(-1, fd, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        // The race lost: another thread published its descriptor first, and
        // may already be using it — close ours, not theirs.
        // SAFETY: `fd` is the descriptor opened above, used nowhere else.
        unsafe { libc::close(fd) };
    }
    THRESHOLD.store(threshold, Ordering::Release);
    threshold
}

/// `PYRUCAST_SPILL_MIN` as a byte count — digits only, no allocation.
fn parse_bytes(digits: &[u8]) -> usize {
    if digits.is_empty() {
        die(b"pyrucast: PYRUCAST_SPILL_MIN must be a byte count\n");
    }
    digits.iter().fold(0usize, |n, &d| {
        if !d.is_ascii_digit() {
            die(b"pyrucast: PYRUCAST_SPILL_MIN must be a byte count\n");
        }
        n.saturating_mul(10).saturating_add((d - b'0') as usize)
    })
}

/// Write `message` to stderr and abort — no allocation, since this runs inside
/// the allocator.
fn die(message: &[u8]) -> ! {
    // SAFETY: a plain write of a borrowed buffer to fd 2.
    unsafe { libc::write(2, message.as_ptr().cast(), message.len()) };
    std::process::abort()
}

/// Whether `layout` lives in a mapping. Called with the same layout on the way
/// in and on the way out, and the threshold never changes once read, so the
/// two calls always agree — no table of mappings to keep.
#[inline]
fn spilled(layout: Layout) -> bool {
    let t = THRESHOLD.load(Ordering::Acquire);
    layout.size() >= t && {
        let t = if t == 0 { configure() } else { t };
        layout.size() >= t && layout.align() <= PAGE
    }
}

/// A fresh, zero-filled shared mapping of `size` bytes, backed by an unnamed
/// file in the spill directory. Null when the file cannot be created or its
/// blocks reserved — the allocation then fails as any other would.
#[cold]
fn map(size: usize) -> *mut u8 {
    let dir = DIR.load(Ordering::Relaxed);
    // SAFETY: plain system calls on descriptors this function owns; every
    // failure is checked before its result is used.
    unsafe {
        let fd = libc::openat(
            dir,
            c".".as_ptr(),
            libc::O_TMPFILE | libc::O_RDWR | libc::O_CLOEXEC,
            0o600,
        );
        if fd < 0 {
            return std::ptr::null_mut();
        }
        // Reserve the blocks now: a sparse file would report a full disk as a
        // SIGBUS on some later write, far from any allocation.
        if libc::fallocate(fd, 0, 0, size as libc::off_t) != 0 {
            libc::close(fd);
            return std::ptr::null_mut();
        }
        let p = libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        );
        // The mapping keeps the file alive; the descriptor is no longer needed.
        libc::close(fd);
        if p == libc::MAP_FAILED {
            return std::ptr::null_mut();
        }
        let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
        PEAK.fetch_max(live, Ordering::Relaxed);
        LARGEST.fetch_max(size, Ordering::Relaxed);
        COUNT.fetch_add(1, Ordering::Relaxed);
        log(b'+', size, live);
        p.cast()
    }
}

/// Unmap a block [`map`] returned, of that same `size`.
#[cold]
fn unmap(ptr: *mut u8, size: usize) {
    // SAFETY: `ptr` is the start of a mapping of `size` bytes (the caller's
    // contract, relayed from `dealloc`).
    unsafe { libc::munmap(ptr.cast(), size) };
    let live = LIVE.fetch_sub(size, Ordering::Relaxed) - size;
    log(b'-', size, live);
}

/// `pyrucast spill: +<size> bytes, <live> mapped` on stderr, when asked for —
/// formatted on the stack, since this runs inside the allocator.
fn log(sign: u8, size: usize, live: usize) {
    if !LOG.load(Ordering::Relaxed) {
        return;
    }
    let mut line = [0u8; 96];
    let mut n = 0;
    let (mut a, mut b) = ([0u8; 20], [0u8; 20]);
    for part in [
        &b"pyrucast spill: "[..],
        &[sign],
        decimal(size, &mut a),
        b" bytes, ",
        decimal(live, &mut b),
        b" mapped\n",
    ] {
        line[n..n + part.len()].copy_from_slice(part);
        n += part.len();
    }
    // SAFETY: a plain write of a stack buffer to fd 2.
    unsafe { libc::write(2, line.as_ptr().cast(), n) };
}

/// `v` in decimal, written at the end of `buf`.
fn decimal(mut v: usize, buf: &mut [u8; 20]) -> &[u8] {
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            return &buf[i..];
        }
    }
}

// SAFETY: every block is either the system allocator's — handed back to it —
// or a page-aligned mapping of at least `layout.size()` bytes, unmapped with
// that same size; `spilled` tells the two apart identically on both sides.
unsafe impl GlobalAlloc for SpillAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if spilled(layout) {
            map(layout.size())
        } else {
            // SAFETY: forwarded under the caller's contract.
            unsafe { System.alloc(layout) }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if spilled(layout) {
            // A freshly allocated file reads as zeros.
            map(layout.size())
        } else {
            // SAFETY: forwarded under the caller's contract.
            unsafe { System.alloc_zeroed(layout) }
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if spilled(layout) {
            unmap(ptr, layout.size());
        } else {
            // SAFETY: forwarded under the caller's contract.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees `new_size`, rounded up to the alignment,
        // does not overflow — the layout is valid.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        if !spilled(layout) && !spilled(new_layout) {
            // SAFETY: both sides are the system allocator's.
            return unsafe { System.realloc(ptr, layout, new_size) };
        }
        // Crossing the threshold, or resizing a mapping: move to a fresh block.
        // A growing `Vec` doubles, so the copies stay amortized.
        // SAFETY: `new_layout` is valid (above).
        let new = unsafe { self.alloc(new_layout) };
        if !new.is_null() {
            // SAFETY: both blocks hold at least the smaller of the two sizes,
            // and a fresh block never overlaps a live one.
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
        }
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_counts_parse_as_decimal() {
        assert_eq!(parse_bytes(b"4096"), 4096);
        assert_eq!(parse_bytes(b"67108864"), DEFAULT_THRESHOLD);
    }

    #[test]
    fn decimals_are_written_without_allocating() {
        let mut buf = [0u8; 20];
        assert_eq!(decimal(0, &mut buf), b"0");
        assert_eq!(decimal(1_073_741_824, &mut buf), b"1073741824");
        assert_eq!(decimal(usize::MAX, &mut buf), b"18446744073709551615");
    }
}
