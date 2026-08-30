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

use crate::containers::matrix::{AssemblyPattern, BlockSlots, Matrix, NamedDof};
use crate::error::{PyrucastError, Result};
use crate::models::kernel;
use crate::ops::coloring;
use crate::parallel::*;
use nalgebra_sparse::CsrMatrix;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;

/// Build the global CSR sparsity [`AssemblyPattern`] for `k` from its blocks'
/// topology alone — no kernel evaluation. Each block contributes its global
/// `(row, col)` entries (a computed block via [`kernel::element_block_pattern`],
/// a literal block via its stored COO indices); columns are then sorted and
/// deduplicated per row.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::MatrixKind;
/// # use pyrucast::ops::{element_field, matrix, mesh, scatter};
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap();
/// # let materiaux = element_field::material_field(&modele,
/// #     &[("k", 1.0), ("rho", 2.0), ("cp", 3.0)]).unwrap();
/// // Le motif est **matériau-indépendant** : il se bâtit une fois, puis
/// // les deux phases numériques le remplissent — la série, référence
/// // bit-à-bit, et la parallèle par coloriage.
/// let k = matrix::stiffness(&modele, &materiaux)?;
/// let motif = scatter::build_pattern(&k)?;
/// let serie = scatter::scatter_serial(&k, &motif)?;
/// let para = scatter::scatter_parallel(&k, &motif)?;
/// assert_eq!(serie.nnz(), motif.nnz());
/// assert_eq!(para.nnz(), serie.nnz());
/// // Mêmes valeurs, à l'ordre de sommation des couleurs près.
/// for (a, b) in serie.values().iter().zip(para.values()) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
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
    // Per block (in `subs` order), its global entries as `(r, c)` in the exact
    // order the numeric scatter emits them — kept so the second pass can map
    // each to its CSR slot once, and cache the result on the pattern.
    enum BlockEntries {
        /// One `(r, c)` list per cell (computed block).
        Computed(Vec<Vec<(usize, usize)>>),
        /// One `(r, c)` per COO entry (literal block).
        Literal(Vec<(usize, usize)>),
    }
    let mut block_entries: Vec<BlockEntries> = Vec::new();
    for blk_h in k {
        let blk = blk_h.read();
        let trow: Vec<usize> = blk.row_dofs().iter().map(|d| row_map[d]).collect();
        let tcol: Vec<usize> = blk.col_dofs().iter().map(|d| col_map[d]).collect();
        match blk.recipe() {
            Some(recipe) => {
                // A non-empty `col_fespaces` marks an inter-mesh block: rows and
                // columns are walked on two facing connectivities instead of one.
                let (_, _, per_cell) = match recipe.col_fespaces.first() {
                    Some(col_fe) => kernel::coupling_block_pattern(
                        &recipe.fespaces[0],
                        col_fe,
                        blk.row_support(),
                        blk.col_support(),
                        blk.dual_vars().len(),
                        blk.primal_vars().len(),
                        blk.ordering(),
                    )?,
                    None => kernel::element_block_pattern(
                        &recipe.fespaces[0],
                        blk.row_support(),
                        blk.col_support(),
                        blk.dual_vars().len(),
                        blk.primal_vars().len(),
                        blk.ordering(),
                    )?,
                };
                let cells: Vec<Vec<(usize, usize)>> = per_cell
                    .iter()
                    .map(|cell| {
                        cell.iter()
                            .map(|&(ri, ci)| {
                                let (gr, gc) = (trow[ri], tcol[ci]);
                                row_cols[gr].push(gc);
                                (gr, gc)
                            })
                            .collect()
                    })
                    .collect();
                block_entries.push(BlockEntries::Computed(cells));
            }
            None => {
                let (lr, lc, _) = blk.local_coo_arrays();
                let entries: Vec<(usize, usize)> = (0..lr.len())
                    .map(|k| {
                        let (gr, gc) = (trow[lr[k]], tcol[lc[k]]);
                        row_cols[gr].push(gc);
                        (gr, gc)
                    })
                    .collect();
                block_entries.push(BlockEntries::Literal(entries));
            }
        }
    }

    // Sort + dedup each row's columns independently → parallel across rows.
    row_cols
        .par_iter_mut()
        .with_min_len(MIN_PARALLEL_LEN)
        .for_each(|cols| {
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

    let mut pattern = AssemblyPattern {
        row_dofs,
        col_dofs,
        row_offsets,
        col_indices,
        block_slots: Vec::new(),
    };

    // Second pass: map every block entry to its CSR value slot once, so the
    // numeric scatter reads slots directly instead of binary-searching per
    // entry on every assembly. Parallel across blocks (independent output).
    // No `with_min_len` here: an item is a whole block (all its cells, all
    // their entries), so blocks are few and each is heavy — the grain policy
    // counts leaf items and would serialise this loop outright.
    pattern.block_slots = block_entries
        .into_par_iter()
        .map(|be| match be {
            BlockEntries::Computed(cells) => BlockSlots::Computed(
                cells
                    .into_iter()
                    .map(|cell| cell.iter().map(|&(r, c)| pattern.slot(r, c)).collect())
                    .collect(),
            ),
            BlockEntries::Literal(entries) => {
                BlockSlots::Literal(entries.iter().map(|&(r, c)| pattern.slot(r, c)).collect())
            }
        })
        .collect();

    Ok(pattern)
}

/// Assemble `k` into a CSR by scattering each block's contribution into
/// `pattern`'s value slots, **serially in block order** (and, within a block,
/// in cell / COO order). Because every slot accumulates in the same order the
/// triplet stream had, the result is bit-for-bit identical to the triplet path.
/// This is the reference numeric phase; the colour-parallel scatter reuses the
/// same pattern and slots.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::MatrixKind;
/// # use pyrucast::ops::{element_field, matrix, mesh, scatter};
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap();
/// # let materiaux = element_field::material_field(&modele,
/// #     &[("k", 1.0), ("rho", 2.0), ("cp", 3.0)]).unwrap();
/// // Le motif est **matériau-indépendant** : il se bâtit une fois, puis
/// // les deux phases numériques le remplissent — la série, référence
/// // bit-à-bit, et la parallèle par coloriage.
/// let k = matrix::stiffness(&modele, &materiaux)?;
/// let motif = scatter::build_pattern(&k)?;
/// let serie = scatter::scatter_serial(&k, &motif)?;
/// let para = scatter::scatter_parallel(&k, &motif)?;
/// assert_eq!(serie.nnz(), motif.nnz());
/// assert_eq!(para.nnz(), serie.nnz());
/// // Mêmes valeurs, à l'ordre de sommation des couleurs près.
/// for (a, b) in serie.values().iter().zip(para.values()) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn scatter_serial(k: &Matrix, pattern: &AssemblyPattern) -> Result<CsrMatrix<f64>> {
    let nrows = pattern.row_dofs.len();
    let ncols = pattern.col_dofs.len();

    let mut values = vec![0.0f64; pattern.nnz()];
    for (bi, blk_h) in k.into_iter().enumerate() {
        let blk = blk_h.read();
        match blk.recipe() {
            Some(recipe) => {
                // Read the sub-model once (not per cell) so the element kernel
                // stays lock-free while it runs in parallel over cells.
                let sm = recipe.submodel.read();
                let phys = sm.as_kind();
                // A computed block exists only for a physics with an element
                // kernel, and such a physics declares a material FE subspace:
                // the question is settled here, once per block, so the per-cell
                // kernels below receive the field itself.
                let material = recipe.material.as_ref().ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "{}: a computed matrix block needs material data",
                        phys.label()
                    ))
                })?;
                // The element kernels live on `Domain`; a computed block has
                // one by construction, since only a `Domain` declares material.
                let domain = phys.as_domain().ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "{}: a computed matrix block needs an element kernel",
                        phys.label()
                    ))
                })?;
                // Resolved **once for the block**, before the parallel region:
                // the closures below capture the table, so no cell ever matches
                // a component name. The guards are dropped straight away — the
                // drivers take the handles and hold their own.
                let lay = {
                    let mat = material.read();
                    let state = recipe.state.as_ref().map(|h| h.read());
                    domain.element_layout(recipe.kind, &mat, state.as_deref())?
                };
                // Values only: `ke` comes out row-major, which is exactly the
                // order the precomputed slots were built in. Asking for the
                // `(row, col)` of each entry here would be asking for indices
                // the pattern already holds — and then dropping them.
                let (cell_values, ke_len) = if recipe.col_fespaces.is_empty() {
                    kernel::element_block_values_per_cell(
                        &recipe.fespaces,
                        blk.dual_vars().len(),
                        blk.primal_vars().len(),
                        material,
                        recipe.state.as_ref(),
                        |geoms, m, state, ke| {
                            phys.matrix_element(recipe.kind, geoms, m, &lay, state, ke)
                        },
                    )?
                } else {
                    kernel::coupling_block_values_per_cell(
                        &recipe.fespaces,
                        &recipe.col_fespaces,
                        blk.dual_vars().len(),
                        blk.primal_vars().len(),
                        material,
                        |row_geoms, col_geoms, m, ke| {
                            domain.coupling_element(recipe.kind, row_geoms, col_geoms, m, &lay, ke)
                        },
                    )?
                };
                let per_cell = cell_values.chunks(ke_len.max(1));
                let slots = match &pattern.block_slots[bi] {
                    BlockSlots::Computed(s) => s,
                    BlockSlots::Literal(_) => {
                        return Err(PyrucastError::Message(
                            "scatter_serial: pattern/block kind mismatch (computed block)".into(),
                        ))
                    }
                };
                let factor = blk.factor();
                for (cell, cell_slots) in per_cell.zip(slots) {
                    for (&v, &slot) in cell.iter().zip(cell_slots) {
                        values[slot] += v * factor;
                    }
                }
            }
            None => {
                let (_, _, lv) = blk.local_coo_arrays();
                let factor = blk.factor();
                let slots = match &pattern.block_slots[bi] {
                    BlockSlots::Literal(s) => s,
                    BlockSlots::Computed(_) => {
                        return Err(PyrucastError::Message(
                            "scatter_serial: pattern/block kind mismatch (literal block)".into(),
                        ))
                    }
                };
                for (&slot, &v) in slots.iter().zip(lv) {
                    values[slot] += v * factor;
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
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::model::Model;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::models::MatrixKind;
/// # use pyrucast::ops::{element_field, matrix, mesh, scatter};
/// # use pyrucast::ops::model;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let modele = model::heat_conduction(&fes).unwrap();
/// # let materiaux = element_field::material_field(&modele,
/// #     &[("k", 1.0), ("rho", 2.0), ("cp", 3.0)]).unwrap();
/// // Le motif est **matériau-indépendant** : il se bâtit une fois, puis
/// // les deux phases numériques le remplissent — la série, référence
/// // bit-à-bit, et la parallèle par coloriage.
/// let k = matrix::stiffness(&modele, &materiaux)?;
/// let motif = scatter::build_pattern(&k)?;
/// let serie = scatter::scatter_serial(&k, &motif)?;
/// let para = scatter::scatter_parallel(&k, &motif)?;
/// assert_eq!(serie.nnz(), motif.nnz());
/// assert_eq!(para.nnz(), serie.nnz());
/// // Mêmes valeurs, à l'ordre de sommation des couleurs près.
/// for (a, b) in serie.values().iter().zip(para.values()) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn scatter_parallel(k: &Matrix, pattern: &AssemblyPattern) -> Result<CsrMatrix<f64>> {
    let nrows = pattern.row_dofs.len();
    let ncols = pattern.col_dofs.len();

    // f64 values held as bits so the colour-parallel scatter can write them
    // through shared references (see `add_atomic`).
    let values: Vec<AtomicU64> = (0..pattern.nnz()).map(|_| AtomicU64::new(0)).collect();

    for (bi, blk_h) in k.into_iter().enumerate() {
        let blk = blk_h.read();
        match blk.recipe() {
            Some(recipe) => {
                let sm = recipe.submodel.read();
                let phys = sm.as_kind();
                // A computed block exists only for a physics with an element
                // kernel, and such a physics declares a material FE subspace:
                // the question is settled here, once per block, so the per-cell
                // kernels below receive the field itself.
                let material = recipe.material.as_ref().ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "{}: a computed matrix block needs material data",
                        phys.label()
                    ))
                })?;
                // The element kernels live on `Domain`; a computed block has
                // one by construction, since only a `Domain` declares material.
                let domain = phys.as_domain().ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "{}: a computed matrix block needs an element kernel",
                        phys.label()
                    ))
                })?;
                // Resolved **once for the block**, before the parallel region:
                // the closures below capture the table, so no cell ever matches
                // a component name. The guards are dropped straight away — the
                // drivers take the handles and hold their own.
                let lay = {
                    let mat = material.read();
                    let state = recipe.state.as_ref().map(|h| h.read());
                    domain.element_layout(recipe.kind, &mat, state.as_deref())?
                };
                // Element matrices, evaluated in parallel — values only, sliced
                // cell by cell for the colour-driven scatter.
                let (cell_values, ke_len) = if recipe.col_fespaces.is_empty() {
                    kernel::element_block_values_per_cell(
                        &recipe.fespaces,
                        blk.dual_vars().len(),
                        blk.primal_vars().len(),
                        material,
                        recipe.state.as_ref(),
                        |geoms, m, state, ke| {
                            phys.matrix_element(recipe.kind, geoms, m, &lay, state, ke)
                        },
                    )?
                } else {
                    kernel::coupling_block_values_per_cell(
                        &recipe.fespaces,
                        &recipe.col_fespaces,
                        blk.dual_vars().len(),
                        blk.primal_vars().len(),
                        material,
                        |row_geoms, col_geoms, m, ke| {
                            domain.coupling_element(recipe.kind, row_geoms, col_geoms, m, &lay, ke)
                        },
                    )?
                };
                let per_cell = cell_values.chunks(ke_len.max(1));
                let slots = match &pattern.block_slots[bi] {
                    BlockSlots::Computed(s) => s,
                    BlockSlots::Literal(_) => {
                        return Err(PyrucastError::Message(
                            "scatter_parallel: pattern/block kind mismatch (computed block)".into(),
                        ))
                    }
                };

                let factor = blk.factor();
                if recipe.col_fespaces.is_empty() {
                    // Cell colouring (cached on the primary FE subspace): two cells
                    // sharing a node conflict, so one colour's cells touch disjoint DOFs.
                    let fe = recipe.fespaces[0].read();
                    let submesh = fe.submesh();
                    let submesh_g = submesh.read();
                    let conn = submesh_g.connectivity();
                    let n_cells = fe.cell_count()?;
                    let keys_per_cell = conn.len().checked_div(n_cells).unwrap_or(0);
                    let coloring =
                        fe.coloring(|| coloring::greedy_color(n_cells, keys_per_cell, conn));

                    // Scatter colour by colour: within a colour, cells write disjoint
                    // slots ⇒ the parallel atomic stores never race. Slots are
                    // precomputed (indexed cell-for-cell, entry-for-entry).
                    let per_cell: Vec<&[f64]> = per_cell.collect();
                    for color in coloring {
                        color
                            .par_iter()
                            .with_min_len(MIN_PARALLEL_LEN)
                            .for_each(|&cell| {
                                for (&v, &slot) in per_cell[cell].iter().zip(&slots[cell]) {
                                    add_atomic(&values[slot], v * factor);
                                }
                            });
                    }
                } else {
                    // An inter-mesh block writes rows on one mesh and columns on
                    // another: a colouring of *one* connectivity no longer proves
                    // the slots disjoint. Its element matrices are still evaluated
                    // in parallel above; only the scatter is serial, as for a
                    // literal block. An interface carries a boundary mesh's worth
                    // of cells, so this costs nothing worth a two-sided colouring.
                    for (cell_values, cell_slots) in per_cell.zip(slots) {
                        for (&v, &slot) in cell_values.iter().zip(cell_slots) {
                            add_atomic(&values[slot], v * factor);
                        }
                    }
                }
            }
            None => {
                let (_, _, lv) = blk.local_coo_arrays();
                let factor = blk.factor();
                let slots = match &pattern.block_slots[bi] {
                    BlockSlots::Literal(s) => s,
                    BlockSlots::Computed(_) => {
                        return Err(PyrucastError::Message(
                            "scatter_parallel: pattern/block kind mismatch (literal block)".into(),
                        ))
                    }
                };
                for (&slot, &v) in slots.iter().zip(lv) {
                    add_atomic(&values[slot], v * factor);
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
