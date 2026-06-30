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
