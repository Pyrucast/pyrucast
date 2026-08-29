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
//! (indexed / `par_chunks_mut` writes), is an **associative reduction over the
//! same values in the same grouping**, or is a **colour-driven scatter**
//! (the `add_atomic` primitive): cells are partitioned into colours that touch disjoint
//! slots, so each slot accumulates its cells in a fixed order — increasing
//! colour, then cell order within the colour — regardless of
//! `RAYON_NUM_THREADS`. The first two are bit-for-bit identical to a sequential
//! run; the colour-driven scatter is reproducible for any thread count but,
//! summed in colour order rather than cell order, is not bit-for-bit with a
//! naive sequential sum (it backs the global assembly and the nodal scatters —
//! the `Bᵀ` divergence, the internal forces and the distributed flux load). The
//! linear solver is the sole non-deterministic exception — see
//! [`crate::ops::solver`].
//!
//! # Grain size
//!
//! Splitting a tiny workload across threads costs more than it saves. Parallel
//! loops use [`MIN_PARALLEL_LEN`] via `with_min_len` / `with_max_len` so small
//! problems (a handful of cells) run effectively sequentially, with no thread
//! hand-off overhead.

pub use rayon::prelude::*;

use crate::error::Result;
use std::sync::atomic::{AtomicU64, Ordering};

/// Minimum number of leaf items (values, cells, …) a rayon job should cover
/// before it is worth splitting further. Tuned so that single small elements
/// (e.g. one SEG2) stay on one thread, while real meshes fan out.
///
/// Apply with `.with_min_len(MIN_PARALLEL_LEN)` on an indexed parallel
/// iterator (`par_iter`, `par_chunks_mut`, …).
///
/// ```
/// # use pyrucast::parallel;
/// // Découper un travail minuscule coûte plus qu'il ne rapporte : en deçà
/// // de ce grain, une boucle parallèle tourne de fait séquentiellement.
/// assert_eq!(parallel::MIN_PARALLEL_LEN, 256);
/// // S'applique par `.with_min_len(MIN_PARALLEL_LEN)` sur un itérateur indexé.
/// ```
pub const MIN_PARALLEL_LEN: usize = 256;

/// Apply `f` to every element of `buf` **in place**, in parallel, honouring the
/// grain-size policy. Each slot is written exactly once ⇒ result is independent
/// of the thread count. The single primitive behind element-wise field maths
/// (`map_all`, the scalar `+ − × ÷` operators, …).
///
/// ```
/// # use pyrucast::parallel;
/// // Chaque case est écrite **exactement une fois** : le résultat ne
/// // dépend pas du nombre de fils. C'est l'unique primitive derrière les
/// // maths de champ élément par élément.
/// let mut v = vec![1.0, 4.0, 9.0];
/// parallel::map_inplace(&mut v, f64::sqrt);
/// assert_eq!(v, vec![1.0, 2.0, 3.0]);
/// ```
pub fn map_inplace(buf: &mut [f64], f: impl Fn(f64) -> f64 + Sync + Send) {
    buf.par_iter_mut()
        .with_min_len(MIN_PARALLEL_LEN)
        .for_each(|v| *v = f(*v));
}

/// Apply `f` in place to component `ci` (stride `ncomp`, `ci < ncomp`) of a
/// flat component-major buffer, in parallel. Each touched slot is written once.
///
/// ```
/// # use pyrucast::parallel;
/// // Une seule composante d'un tampon entrelacé : ici la deuxième de trois.
/// let mut v = vec![1.0, 10.0, 100.0, 2.0, 20.0, 200.0];
/// parallel::map_component_inplace(&mut v, 3, 1, |x| x * 2.0);
/// assert_eq!(v, vec![1.0, 20.0, 100.0, 2.0, 40.0, 200.0]);
/// ```
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

/// Accumulate `v` into the f64 atomic slot `a` (the value held as its bit
/// pattern). The colour-driven scatters — the global assembly
/// ([`crate::ops::matrix`]) and the nodal scatters ([`colored_scatter`], behind
/// the `Bᵀ` divergence and the distributed flux load) — guarantee that within one
/// colour no two parallel cells touch the same slot, so this load-then-store
/// never races; colours run in sequence behind a rayon barrier, so cross-colour
/// accumulation is ordered. `Relaxed` therefore suffices — on x86 the store is a
/// plain `mov`, so a colour-disjoint scatter costs the same as a non-atomic one.
#[inline]
pub(crate) fn add_atomic(a: &AtomicU64, v: f64) {
    let cur = f64::from_bits(a.load(Ordering::Relaxed));
    a.store((cur + v).to_bits(), Ordering::Relaxed);
}

/// Handle onto a colour-scatter's flat f64 accumulator (held as atomics), handed
/// to the per-cell closure of [`colored_scatter`]. Within one colour the cells
/// touch pairwise-disjoint slots, so [`Scatter::add`] never races.
///
/// ```
/// # use pyrucast::parallel;
/// // L'accumulateur d'un scatter colorié, tenu en atomiques. Dans une
/// // couleur, les mailles touchent des cases **disjointes** : `add` ne
/// // court jamais.
/// let couleurs = vec![vec![0, 1]];
/// let mut v = vec![0.0; 2];
/// parallel::colored_scatter(&mut v, &couleurs, 1, || (), |cell, _s, out| {
///     out.add(cell, 1.0);
///     Ok(())
/// })?;
/// assert_eq!(v, vec![1.0, 1.0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub struct Scatter<'a> {
    values: &'a [AtomicU64],
}

impl Scatter<'_> {
    /// Accumulate `v` into slot `slot` of the shared accumulator.
    ///
    /// ```
    /// # use pyrucast::parallel;
    /// // Les contributions **s'accumulent** dans la case visée — d'une couleur
    /// // à la suivante, jamais au sein d'une même. Trois mailles écrivant la
    /// // même case doivent donc être de trois couleurs : c'est exactement ce
    /// // que le coloriage garantit, et sans quoi les additions se perdraient.
    /// let couleurs = vec![vec![0], vec![1], vec![2]];
    /// let mut v = vec![0.0; 1];
    /// parallel::colored_scatter(&mut v, &couleurs, 1, || (), |_cell, _s, out| {
    ///     out.add(0, 2.0);
    ///     Ok(())
    /// })?;
    /// assert_eq!(v, vec![6.0]); // trois mailles × 2
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    #[inline]
    pub fn add(&self, slot: usize, v: f64) {
        add_atomic(&self.values[slot], v);
    }
}

/// Scatter per-cell contributions into a flat accumulator of `n_slots`, in
/// parallel, **colour by colour**. `coloring` partitions the cells so that one
/// colour's cells touch pairwise-disjoint slots; that colour's cells then run
/// concurrently (grain `min_len`) without racing, the colours running in
/// sequence. `init` builds a **per-thread** scratch value reused across a
/// thread's cells (e.g. a local element buffer — pass `|| ()` when none is
/// needed); `cell(c, scratch, out)` computes cell `c`'s contributions and pushes
/// each into `out` via [`Scatter::add`].
///
/// Determinism: each slot accumulates its cells in a fixed order — increasing
/// colour, then cell order within the colour — independent of the thread count,
/// so the result is reproducible for any `RAYON_NUM_THREADS` (see the module
/// *Determinism* note). It is the shared mechanism behind the `Bᵀ` divergence
/// ([`crate::models::kernel::scatter_to_nodes`], behind the `Bᵀ` divergence,
/// the internal forces and the distributed flux load
/// [`crate::ops::node_field::flux`](fn@crate::ops::node_field::flux)). Returns the
/// accumulator as plain `f64`.
///
/// ```
/// # use pyrucast::parallel;
/// // Les mailles sont partagées en couleurs qui touchent des cases
/// // disjointes ; chaque case additionne donc ses mailles dans un ordre
/// // fixe — couleur croissante, puis ordre des mailles — quel que soit
/// // `RAYON_NUM_THREADS`.
/// let couleurs = vec![vec![0, 2], vec![1]];
/// let mut v = vec![0.0; 3];
/// parallel::colored_scatter(&mut v, &couleurs, 1, || 0usize, |cell, tampon, out| {
///     *tampon += 1; // un état par tâche, réutilisé d'une maille à l'autre
///     out.add(cell, (cell + 1) as f64);
///     Ok(())
/// })?;
/// assert_eq!(v, vec![1.0, 2.0, 3.0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn colored_scatter<S>(
    slots: &mut [f64],
    coloring: &[Vec<usize>],
    min_len: usize,
    init: impl Fn() -> S + Sync + Send,
    cell: impl Fn(usize, &mut S, &Scatter) -> Result<()> + Sync,
) -> Result<()> {
    // The accumulation itself needs shared atomic slots — that is what lets two
    // cells of one colour write side by side without a lock and without
    // `unsafe`. It is the one staging buffer this scatter cannot do without.
    //
    // The **result**, though, lands straight in the caller's own buffer: the
    // caller holds it under its write lock for the whole call, so there is no
    // reason to build a second `Vec<f64>` and copy it over.
    let values: Vec<AtomicU64> = (0..slots.len()).map(|_| AtomicU64::new(0)).collect();
    let out = Scatter { values: &values };
    for color in coloring {
        color
            .par_iter()
            .with_min_len(min_len)
            .try_for_each_init(&init, |scratch, &c| cell(c, scratch, &out))?;
    }
    for (slot, a) in slots.iter_mut().zip(values) {
        *slot = f64::from_bits(a.into_inner());
    }
    Ok(())
}
