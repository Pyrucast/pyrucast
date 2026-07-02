//! Per-physics parallel drivers — the bridge that lifts parallelism **above**
//! the constitutive/element kernels.
//!
//! A physics implementation never mentions rayon, the store, or a lock. It
//! supplies two kinds of **pure, sequential kernels**:
//!
//! - a **point kernel** ([`integrate_pointwise`]): the constitutive law at one
//!   Gauss point — read the deformation (+ `VAR0`) and material there, write the
//!   flux/stress (+ `VAR1`);
//! - an **element kernel** ([`assemble_block`]): the local stiffness matrix of
//!   one cell from its geometry and material. It receives **one [`CellGeom`] per
//!   FE subspace** of the block — a single one for a plain volumetric physics, or
//!   several (sharing one mesh, differing by quadrature) for a multi-quadrature
//!   element such as a shear-deformable beam or a shell.
//!
//! These drivers own the fan-out (rayon), the zero-copy borrowing of store data
//! (read guards held across the parallel region, slices borrowed in place — no
//! per-cell snapshot copies), and the deterministic write-back / scatter. See
//! the book chapter *« Parallélisme »* and *« Ajouter une physique »*.
//!
//! # Determinism
//!
//! [`integrate_pointwise`] writes each output slot exactly once
//! (`par_chunks_mut` over cells). [`assemble_block`] computes cell-local
//! matrices in parallel and scatters them into the COO **serially in cell
//! order**, so the assembled matrix is bit-for-bit identical to a sequential
//! run regardless of `RAYON_NUM_THREADS`.

use crate::containers::element_field::SubElementField;
use crate::containers::field::SubField;
use crate::containers::finite_element_space::{
    build_dn_dx, build_jacobian, jacobian_measure, SubFiniteElementSpace,
};
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{Coords, NodeId, SubMesh};
use crate::containers::node_field::SubNodeField;
use crate::error::{PyrucastError, Result};
use crate::parallel::*;
use crate::store::{read, Handle};
use nalgebra_sparse::CooMatrix;
use std::collections::HashMap;

/// Reference-element data of an FE subspace, snapshotted **once** before the
/// parallel loop (it would otherwise re-read the store on every call — through
/// `element_type()` — and serialise all threads on the global store lock).
/// Shape values, reference derivatives and weights at each Gauss point, plus the
/// fixed dimensions. Shared read-only across threads.
struct RefData {
    n_nodes: usize,
    n_gauss: usize,
    space_dim: usize,
    ref_dim: usize,
    /// Shape values `N_i(ξ_g)` per Gauss point.
    n_ref: Vec<Vec<f64>>,
    /// Reference derivatives `∂N_i/∂ξ_k(ξ_g)` per Gauss point.
    dn_ref: Vec<Vec<f64>>,
    /// Gauss weights.
    weights: Vec<f64>,
}

impl RefData {
    fn snapshot(fe: &SubFiniteElementSpace) -> Result<Self> {
        let n_gauss = fe.gauss_count();
        let mut n_ref = Vec::with_capacity(n_gauss);
        let mut dn_ref = Vec::with_capacity(n_gauss);
        let mut weights = Vec::with_capacity(n_gauss);
        for g in 0..n_gauss {
            n_ref.push(fe.n_at_g(g)?.to_vec());
            dn_ref.push(fe.dn_at_g(g)?.to_vec());
            weights.push(fe.gauss_weight(g)?);
        }
        Ok(Self {
            n_nodes: fe.nodes_per_cell()?,
            n_gauss,
            space_dim: fe.space_dim(),
            ref_dim: fe.ref_dim()?,
            n_ref,
            dn_ref,
            weights,
        })
    }
}

/// Geometry of one cell, computed **without touching the store**: coordinates
/// come from a held `Coords` guard, reference data is shared. Every accessor is
/// a pure local computation, so a kernel may call them while the driver runs the
/// cells in parallel — the kernel sees only this, never rayon or the store.
///
/// The cell's node coordinates are gathered **lazily** (only `dn_dx` / `det_j_w`
/// need them), so a point-local kernel that never asks for geometry (most
/// behaviour integrands) pays nothing. `CellGeom` is created and used within one
/// rayon task, so the interior-mutable cache is single-threaded.
pub struct CellGeom<'a> {
    rd: &'a RefData,
    coords: &'a Coords,
    conn: &'a [NodeId],
    cell_coords: std::cell::RefCell<Option<Vec<f64>>>,
    /// Index of this cell within the FE subspace.
    pub cell: usize,
    /// Nodes per cell.
    pub n_nodes: usize,
    /// Gauss points per cell.
    pub n_gauss: usize,
    /// Spatial dimension.
    pub space_dim: usize,
}

impl<'a> CellGeom<'a> {
    fn new(rd: &'a RefData, coords: &'a Coords, conn: &'a [NodeId], cell: usize) -> Result<Self> {
        Ok(Self {
            rd,
            coords,
            conn,
            cell_coords: std::cell::RefCell::new(None),
            cell,
            n_nodes: rd.n_nodes,
            n_gauss: rd.n_gauss,
            space_dim: rd.space_dim,
        })
    }

    /// Global node ids of this cell, in connectivity order.
    pub fn node_ids(&self) -> &'a [NodeId] {
        &self.conn[self.cell * self.n_nodes..(self.cell + 1) * self.n_nodes]
    }

    /// Coordinates of local node `local` (0-based within the cell). Read straight
    /// from the held `Coords` (no gather).
    pub fn node_coord(&self, local: usize) -> Result<&[f64]> {
        self.coords.coord(self.node_ids()[local])
    }

    /// Fill the lazy `cell_coords` cache on first use (gather from the held
    /// `Coords`, no store access).
    fn ensure_cell_coords(&self) -> Result<()> {
        let mut cc = self.cell_coords.borrow_mut();
        if cc.is_none() {
            let mut v = Vec::with_capacity(self.n_nodes * self.space_dim);
            for &id in self.node_ids() {
                v.extend_from_slice(self.coords.coord(id)?);
            }
            *cc = Some(v);
        }
        Ok(())
    }

    /// `∂N_i/∂x_a` at Gauss point `g`, flat layout `[i * space_dim + a]`.
    pub fn dn_dx(&self, g: usize) -> Result<Vec<f64>> {
        self.ensure_cell_coords()?;
        let cc = self.cell_coords.borrow();
        let cc = cc.as_ref().unwrap();
        let dn = &self.rd.dn_ref[g];
        let jac = build_jacobian(cc, dn, self.space_dim, self.rd.ref_dim, self.n_nodes);
        build_dn_dx(&jac, dn, self.space_dim, self.rd.ref_dim, self.n_nodes)
    }

    /// Shape-function values `N_i(ξ_g)` at Gauss point `g`.
    pub fn n_at_g(&self, g: usize) -> Result<&[f64]> {
        Ok(&self.rd.n_ref[g])
    }

    /// `|J|_g · w_g` — the integration weight of Gauss point `g`.
    pub fn det_j_w(&self, g: usize) -> Result<f64> {
        self.ensure_cell_coords()?;
        let cc = self.cell_coords.borrow();
        let cc = cc.as_ref().unwrap();
        let dn = &self.rd.dn_ref[g];
        let jac = build_jacobian(cc, dn, self.space_dim, self.rd.ref_dim, self.n_nodes);
        Ok(jacobian_measure(&jac, self.space_dim, self.rd.ref_dim) * self.rd.weights[g])
    }
}

/// Integrate a point-local constitutive law over `fespace`, in parallel.
///
/// `point(geom, input, material, g, out)` is a pure sequential kernel: for the
/// cell `geom.cell` at Gauss point `g`, it reads the deformation (+ `VAR0`) from
/// `input` and the material from `material` (both borrowed in place), and writes
/// the `out_components.len()` output values into `out`. `material` is `Some` iff
/// the physics declared a material FE subspace.
///
/// Returns the material-state field (flux/stress + `VAR1`) on `fespace`.
pub fn integrate_pointwise(
    fespace: &Handle<SubFiniteElementSpace>,
    input: &Handle<SubElementField>,
    material: Option<&Handle<SubElementField>>,
    out_components: Vec<String>,
    point: impl Fn(&CellGeom, &SubElementField, Option<&SubElementField>, usize, &mut [f64]) -> Result<()>
        + Sync,
) -> Result<SubElementField> {
    let out_stride = out_components.len();
    let mut out = SubElementField::new(fespace.clone(), out_components)?;

    // Guards held for the whole parallel region — slices borrowed, not copied.
    // Reference data snapshotted once (no per-cell store reads inside the loop).
    let fe = read(fespace)?;
    let submesh = fe.submesh();
    let sm = read(&submesh)?;
    let coords_h = sm.coords();
    let coords = read(&coords_h)?;
    let fin = read(input)?;
    let mat_guard = material.map(read).transpose()?;

    let rd = RefData::snapshot(&fe)?;
    let n_gauss = rd.n_gauss;
    let conn: &[NodeId] = sm.connectivity();
    let rd_ref: &RefData = &rd;
    let coords_ref: &Coords = &coords;
    let in_ref: &SubElementField = &fin;
    let mat_ref: Option<&SubElementField> = mat_guard.as_deref();

    out.values_mut()
        .par_chunks_mut(n_gauss * out_stride)
        .with_min_len((MIN_PARALLEL_LEN / n_gauss.max(1)).max(1))
        .enumerate()
        .try_for_each(|(cell, ochunk)| -> Result<()> {
            let geom = CellGeom::new(rd_ref, coords_ref, conn, cell)?;
            for g in 0..n_gauss {
                let slot = &mut ochunk[g * out_stride..(g + 1) * out_stride];
                point(&geom, in_ref, mat_ref, g, slot)?;
            }
            Ok(())
        })?;
    Ok(out)
}

/// Assemble one stiffness block by a per-cell element-matrix kernel, in
/// parallel.
///
/// `element(geoms, material, ke)` is a pure sequential kernel: it fills `ke` —
/// the cell's local dense matrix, row-major, **node-major / variable-minor**:
///   row `r = li * n_dual + di`, col `c = lj * n_primal + pj`
/// (with `li/lj` local node indices, `di/pj` indices into `dual_vars` /
/// `primal_vars`). `geoms` holds one [`CellGeom`] per FE subspace of `fespaces`
/// (same order); a single-space physics reads `geoms[0]`, a multi-quadrature one
/// reads each. The driver scatters `ke` into the block's COO serially in cell
/// order. `material` is `Some` iff the physics supplied one.
#[allow(clippy::too_many_arguments)]
pub fn assemble_block(
    fespaces: &[Handle<SubFiniteElementSpace>],
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    dual_vars: Vec<String>,
    primal_vars: Vec<String>,
    ordering: DofOrdering,
    symmetric: bool,
    material: Option<&Handle<SubElementField>>,
    element: impl Fn(&[CellGeom], Option<&SubElementField>, &mut [f64]) -> Result<()> + Sync,
) -> Result<SubMatrix> {
    let (nrows, ncols, trips) = element_block_triplets(
        fespaces,
        row_support,
        col_support,
        dual_vars.len(),
        primal_vars.len(),
        ordering,
        material,
        element,
    )?;
    let mut rows = Vec::with_capacity(trips.len());
    let mut cols = Vec::with_capacity(trips.len());
    let mut vals = Vec::with_capacity(trips.len());
    for (r, c, v) in trips {
        rows.push(r);
        cols.push(c);
        vals.push(v);
    }
    let coo = CooMatrix::try_from_triplets(nrows, ncols, rows, cols, vals)
        .map_err(|e| PyrucastError::Message(format!("assemble_block: invalid COO: {e}")))?;
    SubMatrix::from_coo(
        row_support.clone(),
        col_support.clone(),
        dual_vars,
        primal_vars,
        ordering,
        symmetric,
        coo,
    )
}

/// `(nrows, ncols, local (row, col, value) triplets)` — the shape returned by
/// [`element_block_triplets`].
pub type BlockTriplets = (usize, usize, Vec<(usize, usize, f64)>);

/// Compute one stiffness block's entries as **local** `(row, col, value)`
/// triplets (the block's own numbering, `(node_local, var)` via `ordering`),
/// driving the per-cell `element` kernel in parallel. This is the shared core of
/// both [`assemble_block`] (which wraps these in a [`SubMatrix`]) and the global
/// computed-block assembler (which remaps them to global indices).
///
/// Cells are computed in parallel and their triplets concatenated **in cell
/// order** (within a cell: `li, di, lj, pj`), so the stream is identical to a
/// sequential run — the assembled values are reproducible. Returns
/// `(nrows, ncols, triplets)`.
#[allow(clippy::too_many_arguments)]
pub fn element_block_triplets(
    fespaces: &[Handle<SubFiniteElementSpace>],
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    n_dual: usize,
    n_primal: usize,
    ordering: DofOrdering,
    material: Option<&Handle<SubElementField>>,
    element: impl Fn(&[CellGeom], Option<&SubElementField>, &mut [f64]) -> Result<()> + Sync,
) -> Result<BlockTriplets> {
    let (nrows, ncols, per_cell) = element_block_triplets_per_cell(
        fespaces,
        row_support,
        col_support,
        n_dual,
        n_primal,
        ordering,
        material,
        element,
    )?;
    let total: usize = per_cell.iter().map(|v| v.len()).sum();
    let mut trips = Vec::with_capacity(total);
    for cell_trips in per_cell {
        trips.extend(cell_trips);
    }
    Ok((nrows, ncols, trips))
}

/// `(nrows, ncols, per-cell local (row, col, value) triplets)` — the shape
/// returned by [`element_block_triplets_per_cell`].
pub type BlockTripletsPerCell = (usize, usize, Vec<Vec<(usize, usize, f64)>>);

/// Same as [`element_block_triplets`] but keeps each cell's triplets in its own
/// `Vec` **instead of concatenating** them. Cells are still computed in parallel
/// and their triplets ordered `li, di, lj, pj` within a cell, so the per-cell
/// lists are reproducible. The grouping lets a colour-driven scatter process one
/// colour's cells (which touch disjoint DOFs) in parallel without write
/// conflicts.
///
/// `fespaces` is the block's FE subspaces: usually one, but several for a
/// multi-quadrature element. They must **share one submesh** (same connectivity,
/// same coordinates, same nodes-per-cell), differing only by quadrature — the
/// primary (index 0) drives the cell loop and the scatter numbering, the others
/// only add their reference data. The kernel receives one [`CellGeom`] per
/// subspace, in order.
#[allow(clippy::too_many_arguments)]
pub fn element_block_triplets_per_cell(
    fespaces: &[Handle<SubFiniteElementSpace>],
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    n_dual: usize,
    n_primal: usize,
    ordering: DofOrdering,
    material: Option<&Handle<SubElementField>>,
    element: impl Fn(&[CellGeom], Option<&SubElementField>, &mut [f64]) -> Result<()> + Sync,
) -> Result<BlockTripletsPerCell> {
    let primary = fespaces.first().ok_or_else(|| {
        PyrucastError::Message("element_block_triplets_per_cell: no FE subspace".into())
    })?;
    let fe = read(primary)?;
    let submesh = fe.submesh();
    let sm = read(&submesh)?;
    let coords_h = sm.coords();
    let coords = read(&coords_h)?;
    let mat_guard = material.map(read).transpose()?;

    // Reference data of every subspace, snapshotted once (they share the submesh
    // ⇒ one connectivity + coords drive every CellGeom; only quadrature differs).
    let mut rds = Vec::with_capacity(fespaces.len());
    rds.push(RefData::snapshot(&fe)?);
    for h in &fespaces[1..] {
        let f = read(h)?;
        if !f.submesh().same_slot(&submesh) {
            return Err(PyrucastError::Message(
                "element_block_triplets_per_cell: all FE subspaces of a block must share one submesh"
                    .into(),
            ));
        }
        rds.push(RefData::snapshot(&f)?);
    }

    let n_cells = fe.cell_count()?;
    let n_nodes = rds[0].n_nodes;
    let conn: &[NodeId] = sm.connectivity();
    let rds_ref: &[RefData] = &rds;
    let coords_ref: &Coords = &coords;
    let mat_ref: Option<&SubElementField> = mat_guard.as_deref();

    let n_cols_loc = n_nodes * n_primal;
    let ke_len = (n_nodes * n_dual) * n_cols_loc;

    // Support → local position maps (first occurrence wins), and the block's
    // local row/col dimensions — all known up front, so the whole loop runs in
    // parallel (no shared mutation).
    let row_nodes: Vec<NodeId> = read(row_support)?.connectivity().to_vec();
    let col_nodes: Vec<NodeId> = read(col_support)?.connectivity().to_vec();
    let n_row_nodes = row_nodes.len();
    let n_col_nodes = col_nodes.len();
    let nrows = n_row_nodes * n_dual;
    let ncols = n_col_nodes * n_primal;
    let pos_map = |nodes: &[NodeId]| -> HashMap<NodeId, u32> {
        let mut m = HashMap::with_capacity(nodes.len());
        for (i, &n) in nodes.iter().enumerate() {
            m.entry(n).or_insert(i as u32);
        }
        m
    };
    let row_pos = pos_map(&row_nodes);
    let col_pos = pos_map(&col_nodes);

    // Per cell, in parallel: compute the element matrix, then emit its triplets
    // in **local** indices. Order within a cell is li,di,lj,pj; cells are
    // concatenated in order below ⇒ identical triplet stream regardless of
    // thread count (bit-for-bit result).
    let per_cell: Vec<Vec<(usize, usize, f64)>> = (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
        .map(|cell| -> Result<Vec<(usize, usize, f64)>> {
            let geoms: Vec<CellGeom> = rds_ref
                .iter()
                .map(|rd| CellGeom::new(rd, coords_ref, conn, cell))
                .collect::<Result<_>>()?;
            let mut ke = vec![0.0_f64; ke_len];
            element(&geoms, mat_ref, &mut ke)?;

            let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
            let mut rpos = Vec::with_capacity(n_nodes);
            let mut cpos = Vec::with_capacity(n_nodes);
            for &nid in ids {
                rpos.push(*row_pos.get(&nid).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "element_block_triplets: node {nid:?} not in row support"
                    ))
                })? as usize);
                cpos.push(*col_pos.get(&nid).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "element_block_triplets: node {nid:?} not in col support"
                    ))
                })? as usize);
            }

            let mut trips = Vec::with_capacity(ke_len);
            for li in 0..n_nodes {
                for di in 0..n_dual {
                    let r = li * n_dual + di;
                    let ri = ordering.to_index(rpos[li], di, n_row_nodes, n_dual);
                    for lj in 0..n_nodes {
                        for pj in 0..n_primal {
                            let c = lj * n_primal + pj;
                            let ci = ordering.to_index(cpos[lj], pj, n_col_nodes, n_primal);
                            trips.push((ri, ci, ke[r * n_cols_loc + c]));
                        }
                    }
                }
            }
            Ok(trips)
        })
        .collect::<Result<_>>()?;

    Ok((nrows, ncols, per_cell))
}

/// `(nrows, ncols, per-cell block-local (row, col) index pairs)` — the shape
/// returned by [`element_block_pattern`].
pub type BlockPattern = (usize, usize, Vec<Vec<(usize, usize)>>);

/// The **symbolic** structure of a computed stiffness block: for each cell, the
/// block-**local** `(row, col)` index pairs it writes, in the exact order
/// [`element_block_triplets`] emits their values (`li, di, lj, pj`). Carries no
/// geometry and evaluates no kernel — only connectivity + the DOF `ordering` —
/// so an assembler can build the global CSR sparsity pattern (and, from it,
/// per-cell scatter slots) cheaply and cache it, then run the numeric kernel
/// only when values are needed. Returns `(nrows, ncols, per_cell_pairs)`.
pub fn element_block_pattern(
    fespace: &Handle<SubFiniteElementSpace>,
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    n_dual: usize,
    n_primal: usize,
    ordering: DofOrdering,
) -> Result<BlockPattern> {
    let fe = read(fespace)?;
    let submesh = fe.submesh();
    let sm = read(&submesh)?;
    let conn: &[NodeId] = sm.connectivity();
    let n_cells = fe.cell_count()?;
    let n_nodes = conn.len().checked_div(n_cells).unwrap_or(0);

    let row_nodes: Vec<NodeId> = read(row_support)?.connectivity().to_vec();
    let col_nodes: Vec<NodeId> = read(col_support)?.connectivity().to_vec();
    let n_row_nodes = row_nodes.len();
    let n_col_nodes = col_nodes.len();
    let nrows = n_row_nodes * n_dual;
    let ncols = n_col_nodes * n_primal;
    let pos_map = |nodes: &[NodeId]| -> HashMap<NodeId, u32> {
        let mut m = HashMap::with_capacity(nodes.len());
        for (i, &n) in nodes.iter().enumerate() {
            m.entry(n).or_insert(i as u32);
        }
        m
    };
    let row_pos = pos_map(&row_nodes);
    let col_pos = pos_map(&col_nodes);

    let per_cell: Vec<Vec<(usize, usize)>> = (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
        .map(|cell| -> Result<Vec<(usize, usize)>> {
            let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
            let mut rpos = Vec::with_capacity(n_nodes);
            let mut cpos = Vec::with_capacity(n_nodes);
            for &nid in ids {
                rpos.push(*row_pos.get(&nid).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "element_block_pattern: node {nid:?} not in row support"
                    ))
                })? as usize);
                cpos.push(*col_pos.get(&nid).ok_or_else(|| {
                    PyrucastError::Message(format!(
                        "element_block_pattern: node {nid:?} not in col support"
                    ))
                })? as usize);
            }
            let mut pairs = Vec::with_capacity(n_nodes * n_dual * n_nodes * n_primal);
            for li in 0..n_nodes {
                for di in 0..n_dual {
                    let ri = ordering.to_index(rpos[li], di, n_row_nodes, n_dual);
                    for lj in 0..n_nodes {
                        for pj in 0..n_primal {
                            let ci = ordering.to_index(cpos[lj], pj, n_col_nodes, n_primal);
                            pairs.push((ri, ci));
                        }
                    }
                }
            }
            Ok(pairs)
        })
        .collect::<Result<_>>()?;

    Ok((nrows, ncols, per_cell))
}

/// Compute the internal nodal forces `f = ∫ Bᵀ σ dΩ` of one physics (Cast3m
/// `BSIG`), by a per-cell force-vector kernel, and scatter them to a
/// [`SubNodeField`] — the mechanical counterpart of
/// [`crate::ops::field::divergence`](fn@crate::ops::field::divergence) (which is
/// exactly this for a scalar transport `Bᵀ q`).
///
/// `element(geoms, stress, fe)` is a pure sequential kernel: for one cell it
/// fills `fe` — the cell's local force vector, **node-major / variable-minor**
/// (`fe[li * n_dual + di]`, `di` indexing `dual_vars`) — from the cell geometry
/// (one [`CellGeom`] per FE subspace of `fespaces`, same order) and the stress
/// borrowed in place. It never sees rayon, the store, or a lock.
///
/// The force vectors are computed in parallel per cell, then **scattered
/// serially in cell order** into `support`'s nodes — a shared node accumulates
/// several cells' contributions, so the sum runs sequentially in cell order and
/// is bit-for-bit identical regardless of thread count. Returns the
/// [`SubNodeField`] with one component per `dual_vars` on `support`.
pub fn internal_forces(
    fespaces: &[Handle<SubFiniteElementSpace>],
    support: &Handle<SubMesh>,
    dual_vars: Vec<String>,
    stress: &Handle<SubElementField>,
    element: impl Fn(&[CellGeom], &SubElementField, &mut [f64]) -> Result<()> + Sync,
) -> Result<SubNodeField> {
    let n_dual = dual_vars.len();
    let primary = fespaces
        .first()
        .ok_or_else(|| PyrucastError::Message("internal_forces: no FE subspace".into()))?;
    let fe = read(primary)?;
    let submesh = fe.submesh();
    let sm = read(&submesh)?;
    let coords_h = sm.coords();
    let coords = read(&coords_h)?;
    let stress_guard = read(stress)?;

    // Reference data of every subspace, snapshotted once (they share the submesh
    // ⇒ one connectivity + coords drive every CellGeom; only quadrature differs).
    let mut rds = Vec::with_capacity(fespaces.len());
    rds.push(RefData::snapshot(&fe)?);
    for h in &fespaces[1..] {
        let f = read(h)?;
        if !f.submesh().same_slot(&submesh) {
            return Err(PyrucastError::Message(
                "internal_forces: all FE subspaces must share one submesh".into(),
            ));
        }
        rds.push(RefData::snapshot(&f)?);
    }

    let n_cells = fe.cell_count()?;
    let n_nodes = rds[0].n_nodes;
    let conn: &[NodeId] = sm.connectivity();
    let rds_ref: &[RefData] = &rds;
    let coords_ref: &Coords = &coords;
    let stress_ref: &SubElementField = &stress_guard;
    let fe_len = n_nodes * n_dual;

    // Per cell, in parallel: the local force vector (written once, disjoint).
    let per_cell: Vec<Vec<f64>> = (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
        .map(|cell| -> Result<Vec<f64>> {
            let geoms: Vec<CellGeom> = rds_ref
                .iter()
                .map(|rd| CellGeom::new(rd, coords_ref, conn, cell))
                .collect::<Result<_>>()?;
            let mut fe = vec![0.0_f64; fe_len];
            element(&geoms, stress_ref, &mut fe)?;
            Ok(fe)
        })
        .collect::<Result<_>>()?;

    // Scatter serially in cell order: a shared node sums several cells, so the
    // f64 accumulation order is fixed ⇒ bit-for-bit deterministic.
    let mut acc: HashMap<NodeId, Vec<f64>> = HashMap::new();
    for (cell, fe_cell) in per_cell.iter().enumerate() {
        let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
        for (li, &nid) in ids.iter().enumerate() {
            let slot = acc.entry(nid).or_insert_with(|| vec![0.0; n_dual]);
            for di in 0..n_dual {
                slot[di] += fe_cell[li * n_dual + di];
            }
        }
    }

    let mut out = SubNodeField::from_poi1(support, dual_vars.clone())?;
    for (nid, vals) in &acc {
        for (di, name) in dual_vars.iter().enumerate() {
            out.set_value(*nid, name, vals[di])?;
        }
    }
    Ok(out)
}
