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
//!   one cell from its geometry and material.
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
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{Coords, NodeId, SubMesh};
use crate::error::Result;
use crate::parallel::*;
use crate::store::{read, Handle};

/// Geometry of one cell, borrowed from shared read guards held by the driver.
///
/// Every accessor is a pure `&self` read of the FE space / coordinates (neither
/// has interior mutability), so a kernel may call them while the driver runs the
/// cells in parallel. The kernel sees only this — never rayon or the store.
pub struct CellGeom<'a> {
    fe: &'a SubFiniteElementSpace,
    coords: &'a Coords,
    conn: &'a [NodeId],
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
    /// Global node ids of this cell, in connectivity order.
    pub fn node_ids(&self) -> &'a [NodeId] {
        &self.conn[self.cell * self.n_nodes..(self.cell + 1) * self.n_nodes]
    }

    /// Coordinates of local node `local` (0-based within the cell).
    pub fn node_coord(&self, local: usize) -> Result<&[f64]> {
        self.coords.coord(self.node_ids()[local])
    }

    /// `∂N_i/∂x_a` at Gauss point `g`, flat layout `[i * space_dim + a]`.
    pub fn dn_dx(&self, g: usize) -> Result<Vec<f64>> {
        self.fe.dn_dx(self.cell, g)
    }

    /// Shape-function values `N_i(ξ_g)` at Gauss point `g`.
    pub fn n_at_g(&self, g: usize) -> Result<&'a [f64]> {
        self.fe.n_at_g(g)
    }

    /// `|J|_g · w_g` — the integration weight of Gauss point `g`.
    pub fn det_j_w(&self, g: usize) -> Result<f64> {
        Ok(self.fe.det_jacobian(self.cell, g)? * self.fe.gauss_weight(g)?)
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
    let fe = read(fespace)?;
    let submesh = fe.submesh();
    let sm = read(&submesh)?;
    let coords_h = sm.coords();
    let coords = read(&coords_h)?;
    let fin = read(input)?;
    let mat_guard = material.map(read).transpose()?;

    let n_nodes = fe.nodes_per_cell()?;
    let n_gauss = fe.gauss_count();
    let space_dim = fe.space_dim();
    let conn: &[NodeId] = sm.connectivity();
    let fe_ref: &SubFiniteElementSpace = &fe;
    let coords_ref: &Coords = &coords;
    let in_ref: &SubElementField = &fin;
    let mat_ref: Option<&SubElementField> = mat_guard.as_deref();

    out.values_mut()
        .par_chunks_mut(n_gauss * out_stride)
        .with_min_len((MIN_PARALLEL_LEN / n_gauss.max(1)).max(1))
        .enumerate()
        .try_for_each(|(cell, ochunk)| -> Result<()> {
            let geom = CellGeom {
                fe: fe_ref,
                coords: coords_ref,
                conn,
                cell,
                n_nodes,
                n_gauss,
                space_dim,
            };
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
/// `element(geom, material, ke)` is a pure sequential kernel: it fills `ke` —
/// the cell's local dense matrix, row-major, **node-major / variable-minor**:
///   row `r = li * n_dual + di`, col `c = lj * n_primal + pj`
/// (with `li/lj` local node indices, `di/pj` indices into `dual_vars` /
/// `primal_vars`). The driver scatters `ke` into the block's COO serially in
/// cell order. `material` is `Some` iff the physics supplied one.
#[allow(clippy::too_many_arguments)]
pub fn assemble_block(
    fespace: &Handle<SubFiniteElementSpace>,
    row_support: &Handle<SubMesh>,
    col_support: &Handle<SubMesh>,
    dual_vars: Vec<String>,
    primal_vars: Vec<String>,
    ordering: DofOrdering,
    symmetric: bool,
    material: Option<&Handle<SubElementField>>,
    element: impl Fn(&CellGeom, Option<&SubElementField>, &mut [f64]) -> Result<()> + Sync,
) -> Result<SubMatrix> {
    let fe = read(fespace)?;
    let submesh = fe.submesh();
    let sm = read(&submesh)?;
    let coords_h = sm.coords();
    let coords = read(&coords_h)?;
    let mat_guard = material.map(read).transpose()?;

    let n_cells = fe.cell_count()?;
    let n_nodes = fe.nodes_per_cell()?;
    let n_gauss = fe.gauss_count();
    let space_dim = fe.space_dim();
    let conn: &[NodeId] = sm.connectivity();
    let fe_ref: &SubFiniteElementSpace = &fe;
    let coords_ref: &Coords = &coords;
    let mat_ref: Option<&SubElementField> = mat_guard.as_deref();

    let n_dual = dual_vars.len();
    let n_primal = primal_vars.len();
    let n_rows_loc = n_nodes * n_dual;
    let n_cols_loc = n_nodes * n_primal;
    let ke_len = n_rows_loc * n_cols_loc;

    // Cell-local matrices in parallel (no shared mutation).
    let locals: Vec<Vec<f64>> = (0..n_cells)
        .into_par_iter()
        .with_min_len((MIN_PARALLEL_LEN / n_nodes.max(1)).max(1))
        .map(|cell| {
            let geom = CellGeom {
                fe: fe_ref,
                coords: coords_ref,
                conn,
                cell,
                n_nodes,
                n_gauss,
                space_dim,
            };
            let mut ke = vec![0.0_f64; ke_len];
            element(&geom, mat_ref, &mut ke)?;
            Ok(ke)
        })
        .collect::<Result<_>>()?;

    // Serial scatter into the COO — insertion order = cell order ⇒ deterministic.
    let mut block = SubMatrix::new(
        row_support.clone(),
        col_support.clone(),
        dual_vars.clone(),
        primal_vars.clone(),
        ordering,
        symmetric,
    )?;
    for cell in 0..n_cells {
        let ids = &conn[cell * n_nodes..(cell + 1) * n_nodes];
        let ke = &locals[cell];
        for li in 0..n_nodes {
            for di in 0..n_dual {
                let r = li * n_dual + di;
                for lj in 0..n_nodes {
                    for pj in 0..n_primal {
                        let c = lj * n_primal + pj;
                        block.add_entry(
                            ids[li],
                            &dual_vars[di],
                            ids[lj],
                            &primal_vars[pj],
                            ke[r * n_cols_loc + c],
                        )?;
                    }
                }
            }
        }
    }
    Ok(block)
}
