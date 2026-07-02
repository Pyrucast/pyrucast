//! Global assembly by **scatter into a fixed CSR pattern**.
//!
//! The historical path turned every block into global `(row, col, value)`
//! triplets and rebuilt a CSR from them (a counting-sort with serial
//! histogram / scatter / dedup passes — the assembly bottleneck). This module
//! splits that into two phases.
//!
//! **Symbolic** ([`build_pattern`]): the global CSR sparsity (`row_offsets` +
//! `col_indices`) from the blocks' topology alone — no kernel eval. It is a
//! function of the model structure only, so it can be built once and reused
//! across assemblies (materials change, sparsity does not).
//!
//! **Numeric** ([`scatter_serial`]): evaluate each block's contribution and add
//! it into the CSR values at its precomputed slot.
//!
//! [`scatter_serial`] visits blocks — and, within a block, cells / COO entries —
//! in the same order the triplet stream had, accumulating each slot in that
//! order, so its result is **bit-for-bit** identical to the triplet path. The
//! parallel colour-driven scatter (the actual speed-up) builds on this same
//! pattern.

use crate::containers::matrix::{AssemblyPattern, Matrix, NamedDof};
use crate::error::{PyrucastError, Result};
use crate::models::kernel;
use crate::ops::assemble::coloring;
use crate::store::read;
use nalgebra_sparse::CsrMatrix;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Build the global CSR sparsity [`AssemblyPattern`] for `k` from its blocks'
/// topology alone — no kernel evaluation. Each block contributes its global
/// `(row, col)` entries (a computed block via [`kernel::element_block_pattern`],
/// a literal block via its stored COO indices); columns are then sorted and
/// deduplicated per row.
pub fn build_pattern(k: &Matrix) -> Result<AssemblyPattern> {
    let row_dofs = k.row_dofs()?;
    let col_dofs = k.col_dofs()?;
    let row_map: HashMap<NamedDof, usize> = row_dofs
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();
    let col_map: HashMap<NamedDof, usize> = col_dofs
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();

    let nrows = row_dofs.len();
    // Per row, the columns it touches (with duplicates; deduped below).
    let mut row_cols: Vec<Vec<usize>> = vec![Vec::new(); nrows];
    for blk_h in k {
        let blk = read(blk_h)?;
        let trow: Vec<usize> = blk.row_dofs().iter().map(|d| row_map[d]).collect();
        let tcol: Vec<usize> = blk.col_dofs().iter().map(|d| col_map[d]).collect();
        match blk.recipe() {
            Some(recipe) => {
                let (_, _, per_cell) = kernel::element_block_pattern(
                    &recipe.fespaces[0],
                    blk.row_support(),
                    blk.col_support(),
                    blk.dual_vars().len(),
                    blk.primal_vars().len(),
                    blk.ordering(),
                )?;
                for cell in &per_cell {
                    for &(ri, ci) in cell {
                        row_cols[trow[ri]].push(tcol[ci]);
                    }
                }
            }
            None => {
                let (lr, lc, _) = blk.local_coo_arrays();
                for k in 0..lr.len() {
                    row_cols[trow[lr[k]]].push(tcol[lc[k]]);
                }
            }
        }
    }

    // Sort + dedup each row's columns independently → parallel across rows.
    row_cols.par_iter_mut().for_each(|cols| {
        cols.sort_unstable();
        cols.dedup();
    });
    // Concatenate into the CSR arrays (serial: O(nnz) appends + prefix sum).
    let mut row_offsets = vec![0usize; nrows + 1];
    let mut col_indices: Vec<usize> = Vec::new();
    for r in 0..nrows {
        col_indices.extend_from_slice(&row_cols[r]);
        row_offsets[r + 1] = col_indices.len();
    }

    Ok(AssemblyPattern {
        row_dofs,
        col_dofs,
        row_offsets,
        col_indices,
    })
}

/// Assemble `k` into a CSR by scattering each block's contribution into
/// `pattern`'s value slots, **serially in block order** (and, within a block,
/// in cell / COO order). Because every slot accumulates in the same order the
/// triplet stream had, the result is bit-for-bit identical to the triplet path.
/// This is the reference numeric phase; the colour-parallel scatter reuses the
/// same pattern and slots.
pub fn scatter_serial(k: &Matrix, pattern: &AssemblyPattern) -> Result<CsrMatrix<f64>> {
    let nrows = pattern.row_dofs.len();
    let ncols = pattern.col_dofs.len();
    let row_map: HashMap<&NamedDof, usize> = pattern
        .row_dofs
        .iter()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();
    let col_map: HashMap<&NamedDof, usize> = pattern
        .col_dofs
        .iter()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();

    let mut values = vec![0.0f64; pattern.nnz()];
    for blk_h in k {
        let blk = read(blk_h)?;
        let brd = blk.row_dofs();
        let bcd = blk.col_dofs();
        let trow: Vec<usize> = brd.iter().map(|d| row_map[d]).collect();
        let tcol: Vec<usize> = bcd.iter().map(|d| col_map[d]).collect();
        match blk.recipe() {
            Some(recipe) => {
                // Read the sub-model once (not per cell) so the element kernel
                // stays lock-free while it runs in parallel over cells.
                let sm = read(&recipe.submodel)?;
                let phys = sm.as_physics();
                let (_, _, trips) = kernel::element_block_triplets(
                    &recipe.fespaces,
                    blk.row_support(),
                    blk.col_support(),
                    blk.dual_vars().len(),
                    blk.primal_vars().len(),
                    blk.ordering(),
                    recipe.material.as_ref(),
                    |geoms, m, ke| phys.element_matrix(geoms, m, ke),
                )?;
                for (ri, ci, v) in trips {
                    values[pattern.slot(trow[ri], tcol[ci])] += v;
                }
            }
            None => {
                let (lr, lc, lv) = blk.local_coo_arrays();
                for k in 0..lv.len() {
                    values[pattern.slot(trow[lr[k]], tcol[lc[k]])] += lv[k];
                }
            }
        }
    }

    CsrMatrix::try_from_csr_data(
        nrows,
        ncols,
        pattern.row_offsets.clone(),
        pattern.col_indices.clone(),
        values,
    )
    .map_err(|e| PyrucastError::Message(format!("scatter_serial: invalid CSR: {e}")))
}

/// Accumulate `v` into the atomic slot `a`. The caller guarantees that, within
/// one colour, no two parallel cells touch the same slot (coloured cells share
/// no DOF), so this load-then-store is never a data race; colours run in
/// sequence, so accumulation across colours is ordered by the rayon barrier
/// between them. `Relaxed` therefore suffices — on x86 it is a plain `mov`, so
/// the colour-disjoint scatter costs the same as a non-atomic one.
#[inline]
fn add_atomic(a: &AtomicU64, v: f64) {
    let cur = f64::from_bits(a.load(Ordering::Relaxed));
    a.store((cur + v).to_bits(), Ordering::Relaxed);
}

/// Assemble `k` into a CSR by scattering each block's contribution into
/// `pattern`'s value slots **in parallel**, colour by colour. A computed
/// block's element matrices are evaluated in parallel
/// ([`kernel::element_block_triplets_per_cell`]); then, for each colour of the
/// block's cell colouring (cached on its FE subspace), that colour's cells —
/// which touch pairwise-disjoint DOFs — scatter concurrently into disjoint
/// slots. Literal blocks scatter serially. The colouring is deterministic, so
/// the assembled values are reproducible regardless of thread count (though the
/// per-slot summation order differs from the serial path, hence not bit-for-bit
/// with it).
pub fn scatter_parallel(k: &Matrix, pattern: &AssemblyPattern) -> Result<CsrMatrix<f64>> {
    let nrows = pattern.row_dofs.len();
    let ncols = pattern.col_dofs.len();
    let row_map: HashMap<&NamedDof, usize> = pattern
        .row_dofs
        .iter()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();
    let col_map: HashMap<&NamedDof, usize> = pattern
        .col_dofs
        .iter()
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect();

    // f64 values held as bits so the colour-parallel scatter can write them
    // through shared references (see `add_atomic`).
    let values: Vec<AtomicU64> = (0..pattern.nnz()).map(|_| AtomicU64::new(0)).collect();

    for blk_h in k {
        let blk = read(blk_h)?;
        let trow: Vec<usize> = blk.row_dofs().iter().map(|d| row_map[d]).collect();
        let tcol: Vec<usize> = blk.col_dofs().iter().map(|d| col_map[d]).collect();
        match blk.recipe() {
            Some(recipe) => {
                let sm = read(&recipe.submodel)?;
                let phys = sm.as_physics();
                // Element matrices, evaluated in parallel, one triplet list per
                // cell (grouping needed for the colour-driven scatter).
                let (_, _, per_cell) = kernel::element_block_triplets_per_cell(
                    &recipe.fespaces,
                    blk.row_support(),
                    blk.col_support(),
                    blk.dual_vars().len(),
                    blk.primal_vars().len(),
                    blk.ordering(),
                    recipe.material.as_ref(),
                    |geoms, m, ke| phys.element_matrix(geoms, m, ke),
                )?;

                // Cell colouring (cached on the primary FE subspace): two cells
                // sharing a node conflict, so one colour's cells touch disjoint DOFs.
                let fe = read(&recipe.fespaces[0])?;
                let submesh = fe.submesh();
                let submesh_g = read(&submesh)?;
                let conn = submesh_g.connectivity();
                let n_cells = fe.cell_count()?;
                let keys_per_cell = conn.len().checked_div(n_cells).unwrap_or(0);
                let coloring = fe.coloring(|| coloring::greedy_color(n_cells, keys_per_cell, conn));

                // Scatter colour by colour: within a colour, cells write disjoint
                // slots ⇒ the parallel atomic stores never race.
                for color in coloring {
                    color.par_iter().try_for_each(|&cell| -> Result<()> {
                        for &(ri, ci, v) in &per_cell[cell] {
                            add_atomic(&values[pattern.slot(trow[ri], tcol[ci])], v);
                        }
                        Ok(())
                    })?;
                }
            }
            None => {
                let (lr, lc, lv) = blk.local_coo_arrays();
                for i in 0..lv.len() {
                    add_atomic(&values[pattern.slot(trow[lr[i]], tcol[lc[i]])], lv[i]);
                }
            }
        }
    }

    let vals: Vec<f64> = values
        .into_iter()
        .map(|a| f64::from_bits(a.into_inner()))
        .collect();
    CsrMatrix::try_from_csr_data(
        nrows,
        ncols,
        pattern.row_offsets.clone(),
        pattern.col_indices.clone(),
        vals,
    )
    .map_err(|e| PyrucastError::Message(format!("scatter_parallel: invalid CSR: {e}")))
}
