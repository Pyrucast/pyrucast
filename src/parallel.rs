//! Shared-memory parallelism helpers (rayon).
//!
//! pyrucast parallelises its heavy compute loops — per-element assembly,
//! per-Gauss-point behaviour integration, element-wise field maths, VTK export
//! — across CPU cores. This module is the single import point for the rayon
//! parallel-iterator prelude, plus the project-wide grain-size policy.
//!
//! # Where parallelism lives
//!
//! It is deliberately **lifted above** the per-physics kernels: the drivers in
//! [`crate::models::kernel`] and the field/container layers call rayon here, so
//! a physics author writes only sequential point-/cell-local maths and never
//! touches a parallel iterator. See the book chapter *« Parallélisme »*.
//!
//! # Determinism
//!
//! Every parallel region either writes each output slot **exactly once**
//! (indexed / `par_chunks_mut` writes) or is an **associative reduction over
//! the same values in the same grouping**. Results are therefore bit-for-bit
//! identical to a sequential run, regardless of `RAYON_NUM_THREADS` (the linear
//! solver is the sole documented exception — see [`crate::ops::solver`]).
//!
//! # Grain size
//!
//! Splitting a tiny workload across threads costs more than it saves. Parallel
//! loops use [`MIN_PARALLEL_LEN`] via `with_min_len` / `with_max_len` so small
//! problems (a handful of cells) run effectively sequentially, with no thread
//! hand-off overhead.

pub use rayon::prelude::*;

/// Minimum number of leaf items (values, cells, …) a rayon job should cover
/// before it is worth splitting further. Tuned so that single small elements
/// (e.g. one SEG2) stay on one thread, while real meshes fan out.
///
/// Apply with `.with_min_len(MIN_PARALLEL_LEN)` on an indexed parallel
/// iterator (`par_iter`, `par_chunks_mut`, …).
pub const MIN_PARALLEL_LEN: usize = 256;

/// Apply `f` to every element of `buf` **in place**, in parallel, honouring the
/// grain-size policy. Each slot is written exactly once ⇒ result is independent
/// of the thread count. The single primitive behind element-wise field maths
/// (`map_all`, the scalar `+ − × ÷` operators, …).
pub fn map_inplace(buf: &mut [f64], f: impl Fn(f64) -> f64 + Sync + Send) {
    buf.par_iter_mut()
        .with_min_len(MIN_PARALLEL_LEN)
        .for_each(|v| *v = f(*v));
}

/// Apply `f` in place to component `ci` (stride `ncomp`, `ci < ncomp`) of a
/// flat component-major buffer, in parallel. Each touched slot is written once.
pub fn map_component_inplace(
    buf: &mut [f64],
    ncomp: usize,
    ci: usize,
    f: impl Fn(f64) -> f64 + Sync + Send,
) {
    debug_assert!(ncomp > 0 && ci < ncomp);
    buf.par_chunks_mut(ncomp)
        .with_min_len((MIN_PARALLEL_LEN / ncomp).max(1))
        .for_each(|row| row[ci] = f(row[ci]));
}
