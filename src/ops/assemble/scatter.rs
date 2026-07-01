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

use crate::containers::matrix::{Matrix, NamedDof};
use crate::error::{PyrucastError, Result};
use crate::models::kernel;
use crate::store::read;
use nalgebra_sparse::CsrMatrix;
use std::collections::HashMap;

/// The global CSR sparsity pattern plus the DOF numbering it indexes. Purely a
/// function of the model's block structure (not of the material values), hence
/// reusable across assemblies of the same model.
pub struct AssemblyPattern {
    /// Global row DOFs, in CSR row order.
    pub row_dofs: Vec<NamedDof>,
    /// Global column DOFs, in CSR column order.
    pub col_dofs: Vec<NamedDof>,
    /// CSR row offsets, length `row_dofs.len() + 1`.
    pub row_offsets: Vec<usize>,
    /// CSR column indices, sorted within each row, length `row_offsets[nrows]`.
    pub col_indices: Vec<usize>,
}

impl AssemblyPattern {
    /// Value-array slot of global entry `(r, c)`. `c` must be present in row
    /// `r`'s column set (it is, for any entry a block contributes — the pattern
    /// was built from exactly those entries).
    #[inline]
    fn slot(&self, r: usize, c: usize) -> usize {
        let base = self.row_offsets[r];
        let seg = &self.col_indices[base..self.row_offsets[r + 1]];
        base + seg
            .binary_search(&c)
            .expect("scatter: entry (r, c) absent from the CSR pattern")
    }

    /// Number of stored entries (CSR `nnz`).
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }
}

/// Build the global CSR sparsity [`AssemblyPattern`] for `k` from its blocks'
/// topology alone — no kernel evaluation. Each block contributes its global
/// `(row, col)` entries (a computed block via [`kernel::element_block_pattern`],
/// a literal block via its stored COO indices); columns are then sorted and
/// deduplicated per row.
pub fn build_pattern(k: &Matrix) -> Result<AssemblyPattern> {
    let row_dofs = k.row_dofs()?;
    let col_dofs = k.col_dofs()?;
    let row_map: HashMap<NamedDof, usize> =
        row_dofs.iter().cloned().enumerate().map(|(i, d)| (d, i)).collect();
    let col_map: HashMap<NamedDof, usize> =
        col_dofs.iter().cloned().enumerate().map(|(i, d)| (d, i)).collect();

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
                    &recipe.fespace,
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

    let mut row_offsets = vec![0usize; nrows + 1];
    let mut col_indices: Vec<usize> = Vec::new();
    for r in 0..nrows {
        let cols = &mut row_cols[r];
        cols.sort_unstable();
        cols.dedup();
        col_indices.extend_from_slice(cols);
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
    let row_map: HashMap<&NamedDof, usize> =
        pattern.row_dofs.iter().enumerate().map(|(i, d)| (d, i)).collect();
    let col_map: HashMap<&NamedDof, usize> =
        pattern.col_dofs.iter().enumerate().map(|(i, d)| (d, i)).collect();

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
                    &recipe.fespace,
                    blk.row_support(),
                    blk.col_support(),
                    blk.dual_vars().len(),
                    blk.primal_vars().len(),
                    blk.ordering(),
                    recipe.material.as_ref(),
                    |geom, m, ke| phys.element_matrix(geom, m, ke),
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
