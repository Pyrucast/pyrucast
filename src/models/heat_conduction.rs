//! Linear heat-conduction physics — assembly of the cell-wise stiffness
//! `K_ij = ∫ k · ∇N_i · ∇N_j dx`.
//!
//! Primal variable `"T"` (temperature, columns), dual `"q"` (heat flux,
//! rows). The conductivity is read from a [`SubElementField`] component
//! named [`MATERIAL_COMPONENT`].

use crate::containers::mesh::NodeId;
use crate::containers::element_field::SubElementField;
use crate::error::Result;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::SubMatrix;
use crate::store::{with, Handle};

/// Column DOF name (temperature).
pub const PRIMAL_VAR: &str = "T";
/// Row DOF name (heat flux).
pub const DUAL_VAR: &str = "q";
/// Required component on the material `SubElementField` (isotropic
/// conductivity).
pub const MATERIAL_COMPONENT: &str = "k";

/// Assemble the heat-conduction stiffness contribution.
///
/// On each cell of `fespace`'s submesh, at each Gauss point `g`:
///   `K_local[i, j] += k(g) · (∇N_i · ∇N_j)|_g · |J|_g · w_g`
/// and the local block is written into `k` at
///   row = `(NodeId_i, "q")`, col = `(NodeId_j, "T")`.
pub fn assemble_stiffness(
    fespace: &Handle<SubFiniteElementSpace>,
    material: &Handle<SubElementField>,
    k: &mut SubMatrix,
) -> Result<()> {
    // Snapshot everything we need from the FE space and submesh in one
    // pass. We then drop the FE space lock before reading the material
    // (different store type, but better hygiene to keep critical
    // sections small).
    struct CellSnapshot {
        node_ids: Vec<NodeId>,
        dn_dx: Vec<Vec<f64>>, // [g][i * space_dim + a]
        det_j_w: Vec<f64>,    // |J|_g · w_g
    }

    let snapshots: Vec<CellSnapshot> = with(fespace, |s| -> Result<_> {
        let n_cells = s.cell_count()?;
        let n_nodes = s.nodes_per_cell()?;
        let n_g = s.gauss_count();
        let submesh = s.submesh();

        let conn: Vec<NodeId> = with(&submesh, |sm| sm.connectivity().to_vec())?;

        let mut out = Vec::with_capacity(n_cells);
        for cell in 0..n_cells {
            let ids = conn[cell * n_nodes..(cell + 1) * n_nodes].to_vec();
            let mut dn_dx: Vec<Vec<f64>> = Vec::with_capacity(n_g);
            let mut det_j_w: Vec<f64> = Vec::with_capacity(n_g);
            for g in 0..n_g {
                dn_dx.push(s.dn_dx(cell, g)?);
                det_j_w.push(s.det_jacobian(cell, g)? * s.gauss_weight(g)?);
            }
            out.push(CellSnapshot {
                node_ids: ids,
                dn_dx,
                det_j_w,
            });
        }
        Ok(out)
    })??;

    let space_dim = with(fespace, |s| s.space_dim())?;
    let n_nodes = with(fespace, |s| s.nodes_per_cell())??;
    let n_g = with(fespace, |s| s.gauss_count())?;

    // Read material conductivity once per (cell, gauss).
    let mut conductivities: Vec<Vec<f64>> = Vec::with_capacity(snapshots.len());
    with(material, |f| -> Result<()> {
        for cell in 0..snapshots.len() {
            let mut row = Vec::with_capacity(n_g);
            for g in 0..n_g {
                row.push(f.value(cell, g, MATERIAL_COMPONENT)?);
            }
            conductivities.push(row);
        }
        Ok(())
    })??;

    // Assemble cell by cell.
    for (cell, snap) in snapshots.iter().enumerate() {
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let mut k_ij = 0.0;
                for g in 0..n_g {
                    let mut grad_dot = 0.0;
                    for a in 0..space_dim {
                        grad_dot += snap.dn_dx[g][i * space_dim + a]
                            * snap.dn_dx[g][j * space_dim + a];
                    }
                    k_ij += conductivities[cell][g] * grad_dot * snap.det_j_w[g];
                }
                k.add_entry(snap.node_ids[i], DUAL_VAR, snap.node_ids[j], PRIMAL_VAR, k_ij)?;
            }
        }
    }
    Ok(())
}
