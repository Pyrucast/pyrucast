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
//! it into the CSR values at its precomputed slot. Both numeric phases yield
//! the **values alone**, in the pattern's slot order: the sparsity is the
//! pattern's, and the assembled matrix shares it rather than copying it.
//!
//! [`scatter_serial`] visits blocks — and, within a block, cells / COO entries —
//! in the same order the triplet stream had, accumulating each slot in that
//! order, so its result is **bit-for-bit** identical to the triplet path. The
//! parallel colour-driven scatter (the actual speed-up) builds on this same
//! pattern.

use crate::containers::matrix::KernelInputs;
use crate::containers::matrix::{
    dof_node, dof_var, AssemblyPattern, BlockSlots, DofKey, DofOrdering, Matrix,
};
use crate::error::{PyrucastError, Result};
use crate::models::kernel;
use crate::models::KernelState;
use crate::parallel::*;

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;

/// DOF key → global index, for the one lookup the symbolic phase does per block
/// DOF.
///
/// A `HashMap<(NodeId, String), usize>` was thirty million string hashes on a
/// solid mesh. Node ids index `Coords` directly, so `node × n_vars + var` is a
/// perfect hash — a plain slice indexed in constant time, with no hashing at
/// all. It only stops paying when the id space is far larger than the DOF set
/// it holds (a `Coords` riddled with deletions), and there the compact `u64`
/// keys still hash for a fraction of a string's cost.
enum DofIndex {
    Dense { at: Vec<u32>, n_vars: usize },
    Sparse(HashMap<DofKey, usize>),
}

impl DofIndex {
    /// Index `keys` by their position, `n_vars` naming the variable table's width.
    fn build(keys: &[DofKey], n_vars: usize) -> Self {
        let n_vars = n_vars.max(1);
        let max_node = keys.iter().map(|&k| dof_node(k).0).max().unwrap_or(0);
        let slots = (max_node as usize + 1).saturating_mul(n_vars);
        if slots <= keys.len().saturating_mul(8).max(1024) {
            let mut at = vec![u32::MAX; slots];
            for (i, &k) in keys.iter().enumerate() {
                at[dof_node(k).0 as usize * n_vars + dof_var(k) as usize] = i as u32;
            }
            DofIndex::Dense { at, n_vars }
        } else {
            DofIndex::Sparse(keys.iter().enumerate().map(|(i, &k)| (k, i)).collect())
        }
    }

    /// The global index of `key`, which the caller's block is known to carry.
    #[inline]
    fn get(&self, key: DofKey) -> Result<usize> {
        let found = match self {
            DofIndex::Dense { at, n_vars } => at
                .get(dof_node(key).0 as usize * *n_vars + dof_var(key) as usize)
                .copied()
                .filter(|&i| i != u32::MAX)
                .map(|i| i as usize),
            DofIndex::Sparse(map) => map.get(&key).copied(),
        };
        found.ok_or_else(|| {
            PyrucastError::Message(format!(
                "build_pattern: DOF (node {:?}, variable slot {}) is absent from \
                 the global numbering",
                dof_node(key),
                dof_var(key)
            ))
        })
    }
}

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
/// assert_eq!(serie.len(), motif.nnz());
/// assert_eq!(para.len(), serie.len());
/// // Mêmes valeurs, à l'ordre de sommation des couleurs près.
/// for (a, b) in serie.iter().zip(&para) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn build_pattern(k: &Matrix) -> Result<AssemblyPattern> {
    let vars = k.dof_vars()?;
    let row_keys = k.row_dof_keys()?;
    let col_keys = k.col_dof_keys()?;
    let slot_of: HashMap<String, u32> = vars
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i as u32))
        .collect();
    let row_map = DofIndex::build(&row_keys, vars.len());
    let col_map = DofIndex::build(&col_keys, vars.len());

    let nrows = row_keys.len();

    // Per block, the compact shape the two passes below walk: a computed block
    // is its [`kernel::BlockPattern`] plus the local→global DOF translation; a
    // literal one is its global `(r, c)` entries.
    enum BlockShape {
        Computed {
            pat: kernel::BlockPattern,
            trow: Vec<u32>,
            tcol: Vec<u32>,
        },
        Literal(Vec<(u32, u32)>),
    }
    let mut shapes: Vec<BlockShape> = Vec::new();
    for blk_h in k {
        let blk = blk_h.read();
        let trow: Vec<u32> = blk
            .row_dof_keys(&slot_of)?
            .iter()
            .map(|&d| row_map.get(d).map(|i| i as u32))
            .collect::<Result<_>>()?;
        let tcol: Vec<u32> = blk
            .col_dof_keys(&slot_of)?
            .iter()
            .map(|&d| col_map.get(d).map(|i| i as u32))
            .collect::<Result<_>>()?;
        shapes.push(match blk.recipe() {
            Some(recipe) => {
                // A non-empty `col_fespaces` marks an inter-mesh block: rows and
                // columns are walked on two facing connectivities instead of one.
                let pat = match recipe.col_fespaces.first() {
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
                BlockShape::Computed { pat, trow, tcol }
            }
            None => {
                let (lr, lc, _) = blk.local_coo_arrays();
                BlockShape::Literal((0..lr.len()).map(|k| (trow[lr[k]], tcol[lc[k]])).collect())
            }
        });
    }

    // ── Pass 1: how many columns each row is handed (duplicates included) ───
    //
    // A computed block hands row `(li, di)` of a cell one run of
    // `col_nodes × n_primal` columns, so counting needs no column enumerated at
    // all. This is what a `Vec<Vec<usize>>` — one growing vector per row, thirty
    // million of them — used to discover the hard way.
    let mut counts = vec![0usize; nrows];
    for shape in &shapes {
        match shape {
            BlockShape::Computed { pat, trow, .. } => {
                let run = pat.col_nodes_per_cell * pat.n_primal;
                for cell in 0..pat.n_cells {
                    for li in 0..pat.row_nodes_per_cell {
                        for di in 0..pat.n_dual {
                            counts[trow[pat.row_index(cell, li, di)] as usize] += run;
                        }
                    }
                }
            }
            BlockShape::Literal(entries) => {
                for &(r, _) in entries {
                    counts[r as usize] += 1;
                }
            }
        }
    }
    let mut starts = vec![0usize; nrows + 1];
    for r in 0..nrows {
        starts[r + 1] = starts[r] + counts[r];
    }

    // ── Pass 2: write the columns into each row's run ───────────────────────
    let mut dup_cols = vec![0u32; starts[nrows]];
    let mut cursor = starts[..nrows].to_vec();
    for shape in &shapes {
        match shape {
            BlockShape::Computed { pat, trow, tcol } => {
                for cell in 0..pat.n_cells {
                    for li in 0..pat.row_nodes_per_cell {
                        for di in 0..pat.n_dual {
                            let r = trow[pat.row_index(cell, li, di)] as usize;
                            let at = cursor[r];
                            let mut w = at;
                            for lj in 0..pat.col_nodes_per_cell {
                                for pj in 0..pat.n_primal {
                                    dup_cols[w] = tcol[pat.col_index(cell, lj, pj)];
                                    w += 1;
                                }
                            }
                            cursor[r] = w;
                        }
                    }
                }
            }
            BlockShape::Literal(entries) => {
                for &(r, c) in entries {
                    dup_cols[cursor[r as usize]] = c;
                    cursor[r as usize] += 1;
                }
            }
        }
    }

    // ── Pass 3: sort + dedup each row's run, then compact ───────────────────
    //
    // Rows are independent, so this fans out; each run is sorted in place inside
    // the one buffer, and `kept` records how much of it survives.
    let mut kept = vec![0usize; nrows];
    {
        let mut rest: &mut [u32] = &mut dup_cols;
        let mut segments: Vec<&mut [u32]> = Vec::with_capacity(nrows);
        for r in 0..nrows {
            let (seg, tail) = rest.split_at_mut(counts[r]);
            segments.push(seg);
            rest = tail;
        }
        segments
            .par_iter_mut()
            .with_min_len(MIN_PARALLEL_LEN)
            .zip(kept.par_iter_mut())
            .for_each(|(seg, kept)| {
                seg.sort_unstable();
                let mut n = 0usize;
                for i in 0..seg.len() {
                    if n == 0 || seg[n - 1] != seg[i] {
                        seg[n] = seg[i];
                        n += 1;
                    }
                }
                *kept = n;
            });
    }
    let mut row_offsets = vec![0usize; nrows + 1];
    for r in 0..nrows {
        row_offsets[r + 1] = row_offsets[r] + kept[r];
    }
    let mut col_indices: Vec<usize> = Vec::with_capacity(row_offsets[nrows]);
    for r in 0..nrows {
        col_indices.extend(
            dup_cols[starts[r]..starts[r] + kept[r]]
                .iter()
                .map(|&c| c as usize),
        );
    }
    drop(dup_cols);

    let mut pattern = AssemblyPattern {
        vars,
        row_keys,
        col_keys,
        row_offsets: std::sync::Arc::new(row_offsets),
        col_indices: std::sync::Arc::new(col_indices),
        block_slots: Vec::new(),
    };

    // ── Pass 4: every entry's CSR slot, resolved once and cached ────────────
    //
    // The numeric scatter then reads its slots directly instead of
    // binary-searching per entry on every assembly. Parallel across blocks
    // (independent output). No `with_min_len` here: an item is a whole block —
    // all its cells, all their entries — so blocks are few and each is heavy,
    // and the grain policy, which counts leaf items, would serialise this loop.
    if pattern.nnz() > u32::MAX as usize {
        return Err(PyrucastError::Message(format!(
            "build_pattern: {} stored entries exceed the {} a 32-bit slot index \
             can name — this problem is past what an assembled matrix holds",
            pattern.nnz(),
            u32::MAX
        )));
    }
    pattern.block_slots = shapes
        .into_par_iter()
        .map(|shape| match shape {
            BlockShape::Computed { pat, trow, tcol } => {
                Ok(computed_slots(&pattern, &pat, &trow, &tcol))
            }
            BlockShape::Literal(entries) => Ok(BlockSlots::Literal(
                entries
                    .iter()
                    .map(|&(r, c)| pattern.slot(r as usize, c as usize))
                    .collect(),
            )),
        })
        .collect::<Result<_>>()?;

    Ok(pattern)
}

/// The CSR slots of one computed block, in the kernel's `(li, di, lj, pj)`
/// emission order.
///
/// Takes the **blocked** form when the block's own numbering makes a node's
/// primal columns consecutive — which [`DofOrdering::NodesThenVars`] intends,
/// but which only the global numbering can settle, so it is checked rather than
/// assumed. That check is what lets one base slot stand for `n_primal` of them.
fn computed_slots(
    pattern: &AssemblyPattern,
    pat: &kernel::BlockPattern,
    trow: &[u32],
    tcol: &[u32],
) -> BlockSlots {
    let blocked = pat.n_primal > 1
        && pat.ordering == DofOrdering::NodesThenVars
        && (0..pat.n_col_support).all(|q| {
            let base = tcol[pat.ordering.to_index(q, 0, pat.n_col_support, pat.n_primal)];
            (1..pat.n_primal).all(|pj| {
                tcol[pat
                    .ordering
                    .to_index(q, pj, pat.n_col_support, pat.n_primal)]
                    == base + pj as u32
            })
        });

    let rows_per_cell = pat.row_nodes_per_cell * pat.n_dual;
    if blocked {
        let stride = rows_per_cell * pat.col_nodes_per_cell;
        let mut bases = vec![0u32; pat.n_cells * stride];
        bases
            .par_chunks_mut(stride)
            .with_min_len((MIN_PARALLEL_LEN / stride.max(1)).max(1))
            .enumerate()
            .for_each(|(cell, out)| {
                let mut w = 0;
                for li in 0..pat.row_nodes_per_cell {
                    for di in 0..pat.n_dual {
                        let r = trow[pat.row_index(cell, li, di)] as usize;
                        for lj in 0..pat.col_nodes_per_cell {
                            let c = tcol[pat.col_index(cell, lj, 0)] as usize;
                            out[w] = pattern.slot(r, c) as u32;
                            w += 1;
                        }
                    }
                }
            });
        return BlockSlots::ComputedBlocked {
            bases,
            stride,
            n_primal: pat.n_primal,
        };
    }

    let stride = pat.entries_per_cell();
    let mut slots = vec![0u32; pat.n_cells * stride];
    slots
        .par_chunks_mut(stride.max(1))
        .with_min_len((MIN_PARALLEL_LEN / stride.max(1)).max(1))
        .enumerate()
        .for_each(|(cell, out)| {
            let mut w = 0;
            for li in 0..pat.row_nodes_per_cell {
                for di in 0..pat.n_dual {
                    let r = trow[pat.row_index(cell, li, di)] as usize;
                    for lj in 0..pat.col_nodes_per_cell {
                        for pj in 0..pat.n_primal {
                            let c = tcol[pat.col_index(cell, lj, pj)] as usize;
                            out[w] = pattern.slot(r, c) as u32;
                            w += 1;
                        }
                    }
                }
            }
        });
    BlockSlots::Computed { slots, stride }
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
/// assert_eq!(serie.len(), motif.nnz());
/// assert_eq!(para.len(), serie.len());
/// // Mêmes valeurs, à l'ordre de sommation des couleurs près.
/// for (a, b) in serie.iter().zip(&para) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn scatter_serial(k: &Matrix, pattern: &AssemblyPattern) -> Result<Vec<f64>> {
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
                // Les gardes des entrées vivent le temps du bloc : `inputs` pointe
                // dedans. Celle du matériau, elle, est relâchée aussitôt — les
                // pilotes prennent la poignée et tiennent la leur.
                let state_guard = match &recipe.inputs {
                    KernelInputs::State(h) => Some(h.read()),
                    _ => None,
                };
                let behavior_guards = match &recipe.inputs {
                    KernelInputs::Behavior {
                        deformation, prev, ..
                    } => Some((deformation.read(), prev.read())),
                    _ => None,
                };
                // Résolus **une fois pour le bloc**, avant la région parallèle :
                // les fermetures capturent les tables, donc aucune maille ne
                // compare jamais un nom de composante.
                let (lay, zone) = {
                    let mat = material.read();
                    let zone = match &behavior_guards {
                        Some((def, prev)) => Some(domain.zone_layout(def, prev, &mat)?),
                        None => None,
                    };
                    (
                        domain.element_layout(recipe.kind, &mat, state_guard.as_deref())?,
                        zone,
                    )
                };
                let inputs = match (&recipe.inputs, &state_guard, &behavior_guards, &zone) {
                    (KernelInputs::MaterialOnly, ..) => KernelState::MaterialOnly,
                    (KernelInputs::State(_), Some(g), ..) => KernelState::State(g),
                    (KernelInputs::Behavior { dt, .. }, _, Some((def, prev)), Some(z)) => {
                        KernelState::Behavior {
                            lay: z,
                            deformation: def,
                            prev,
                            dt: *dt,
                        }
                    }
                    _ => {
                        return Err(PyrucastError::Message(format!(
                            "{}: computed block inputs do not match its matrix kind",
                            phys.label()
                        )))
                    }
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
                        |geoms, m, ke| {
                            phys.matrix_element(recipe.kind, geoms, m, &lay, &inputs, ke)
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
                let slots = &pattern.block_slots[bi];
                if matches!(slots, BlockSlots::Literal(_)) {
                    return Err(PyrucastError::Message(
                        "scatter_serial: pattern/block kind mismatch (computed block)".into(),
                    ));
                }
                let factor = blk.factor();
                for (cell, ke) in cell_values.chunks(ke_len.max(1)).enumerate() {
                    slots.each_entry(cell, ke, |slot, v| values[slot] += v * factor);
                }
            }
            None => {
                let (_, _, lv) = blk.local_coo_arrays();
                let factor = blk.factor();
                let BlockSlots::Literal(slots) = &pattern.block_slots[bi] else {
                    return Err(PyrucastError::Message(
                        "scatter_serial: pattern/block kind mismatch (literal block)".into(),
                    ));
                };
                for (&slot, &v) in slots.iter().zip(lv) {
                    values[slot] += v * factor;
                }
            }
        }
    }

    Ok(values)
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
/// assert_eq!(serie.len(), motif.nnz());
/// assert_eq!(para.len(), serie.len());
/// // Mêmes valeurs, à l'ordre de sommation des couleurs près.
/// for (a, b) in serie.iter().zip(&para) {
///     assert!((a - b).abs() < 1e-12);
/// }
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn scatter_parallel(k: &Matrix, pattern: &AssemblyPattern) -> Result<Vec<f64>> {
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
                // Les gardes des entrées vivent le temps du bloc : `inputs` pointe
                // dedans. Celle du matériau, elle, est relâchée aussitôt — les
                // pilotes prennent la poignée et tiennent la leur.
                let state_guard = match &recipe.inputs {
                    KernelInputs::State(h) => Some(h.read()),
                    _ => None,
                };
                let behavior_guards = match &recipe.inputs {
                    KernelInputs::Behavior {
                        deformation, prev, ..
                    } => Some((deformation.read(), prev.read())),
                    _ => None,
                };
                // Résolus **une fois pour le bloc**, avant la région parallèle :
                // les fermetures capturent les tables, donc aucune maille ne
                // compare jamais un nom de composante.
                let (lay, zone) = {
                    let mat = material.read();
                    let zone = match &behavior_guards {
                        Some((def, prev)) => Some(domain.zone_layout(def, prev, &mat)?),
                        None => None,
                    };
                    (
                        domain.element_layout(recipe.kind, &mat, state_guard.as_deref())?,
                        zone,
                    )
                };
                let inputs = match (&recipe.inputs, &state_guard, &behavior_guards, &zone) {
                    (KernelInputs::MaterialOnly, ..) => KernelState::MaterialOnly,
                    (KernelInputs::State(_), Some(g), ..) => KernelState::State(g),
                    (KernelInputs::Behavior { dt, .. }, _, Some((def, prev)), Some(z)) => {
                        KernelState::Behavior {
                            lay: z,
                            deformation: def,
                            prev,
                            dt: *dt,
                        }
                    }
                    _ => {
                        return Err(PyrucastError::Message(format!(
                            "{}: computed block inputs do not match its matrix kind",
                            phys.label()
                        )))
                    }
                };
                let slots = &pattern.block_slots[bi];
                if matches!(slots, BlockSlots::Literal(_)) {
                    return Err(PyrucastError::Message(
                        "scatter_parallel: pattern/block kind mismatch (computed block)".into(),
                    ));
                }
                let factor = blk.factor();

                if recipe.col_fespaces.is_empty() {
                    // Evaluation and scatter in the same breath, colour by
                    // colour: each cell's `ke` is produced on a per-task scratch
                    // and poured straight into its precomputed slots. Within a
                    // colour the cells touch disjoint DOFs, so the atomic stores
                    // never race; the colours run in sequence, so each slot
                    // accumulates in a fixed order whatever the thread count.
                    kernel::element_block_colored(
                        &recipe.fespaces,
                        blk.dual_vars().len(),
                        blk.primal_vars().len(),
                        material,
                        |geoms, m, ke| {
                            phys.matrix_element(recipe.kind, geoms, m, &lay, &inputs, ke)
                        },
                        |cell, ke| {
                            slots.each_entry(cell, ke, |slot, v| {
                                add_atomic(&values[slot], v * factor)
                            });
                            Ok(())
                        },
                    )?;
                } else {
                    // An inter-mesh block writes rows on one mesh and columns on
                    // another: a colouring of *one* connectivity no longer proves
                    // the slots disjoint. Its element matrices are still evaluated
                    // in parallel; only the scatter is serial, as for a literal
                    // block. An interface carries a boundary mesh's worth of
                    // cells, so this costs nothing worth a two-sided colouring.
                    let (cell_values, ke_len) = kernel::coupling_block_values_per_cell(
                        &recipe.fespaces,
                        &recipe.col_fespaces,
                        blk.dual_vars().len(),
                        blk.primal_vars().len(),
                        material,
                        |row_geoms, col_geoms, m, ke| {
                            domain.coupling_element(recipe.kind, row_geoms, col_geoms, m, &lay, ke)
                        },
                    )?;
                    for (cell, ke) in cell_values.chunks(ke_len.max(1)).enumerate() {
                        slots.each_entry(cell, ke, |slot, v| add_atomic(&values[slot], v * factor));
                    }
                }
            }
            None => {
                let (_, _, lv) = blk.local_coo_arrays();
                let factor = blk.factor();
                let BlockSlots::Literal(slots) = &pattern.block_slots[bi] else {
                    return Err(PyrucastError::Message(
                        "scatter_parallel: pattern/block kind mismatch (literal block)".into(),
                    ));
                };
                for (&slot, &v) in slots.iter().zip(lv) {
                    add_atomic(&values[slot], v * factor);
                }
            }
        }
    }

    // `AtomicU64` and `f64` share a layout, so this reuses the allocation rather
    // than raising a second buffer the size of the whole value array.
    Ok(values
        .into_iter()
        .map(|a| f64::from_bits(a.into_inner()))
        .collect())
}
