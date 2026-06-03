//! Linear heat-conduction physics — assembly of the cell-wise stiffness
//! `K_ij = ∫ k · ∇N_i · ∇N_j dx`.
//!
//! Primal variable `"T"` (temperature, columns), dual `"q"` (heat flux,
//! rows). The conductivity is read from a [`SubElementField`] component
//! named [`MATERIAL_COMPONENT`].

use crate::containers::element_field::SubElementField;
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::NodeId;
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::Result;
use crate::models::Physics;
use crate::store::{insert, with, Handle};
use serde::{Deserialize, Serialize};

/// Column DOF name (temperature).
pub const PRIMAL_VAR: &str = "T";
/// Row DOF name (heat flux).
pub const DUAL_VAR: &str = "q";
/// Required component on the material `SubElementField` (isotropic
/// conductivity).
pub const MATERIAL_COMPONENT: &str = "k";
/// Material contract returned by [`Physics::material_components`].
const MATERIAL_COMPONENTS: &[&str] = &[MATERIAL_COMPONENT];

/// Linear heat conduction.
///
/// - primal variable: `"T"` (temperature, columns).
/// - dual variable:   `"q"` (heat flux row labels).
/// - Material data (conductivity `"k"`, …) is **not** stored here; it is
///   supplied at assembly time via [`crate::ops::assemble::stiffness`].
#[derive(Clone, Serialize, Deserialize)]
pub struct HeatConduction {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    /// POI1 SubMesh covering the unique nodes of `fespace`'s submesh,
    /// built once at construction. Reused as the row/col support of every
    /// assembled stiffness block — no per-assembly rebuild.
    pub(crate) support: Handle<SubMesh>,
}

impl HeatConduction {
    /// Heat-conduction physics on an FE subspace. Builds the stable POI1
    /// [`SubMesh`] covering the subspace's unique nodes (reused as the
    /// row/col support of every assembled block).
    pub fn new(fespace: Handle<SubFiniteElementSpace>) -> Result<Self> {
        let submesh = with(&fespace, |s| s.submesh())?;
        let support = insert(with(&submesh, |s| s.to_poi1())??);
        Ok(Self { fespace, support })
    }
}

impl Physics for HeatConduction {
    fn primal_vars(&self) -> Vec<String> {
        vec![PRIMAL_VAR.to_string()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![DUAL_VAR.to_string()]
    }

    fn material_components(&self) -> Option<&'static [&'static str]> {
        Some(MATERIAL_COMPONENTS)
    }

    fn material_fespace(&self) -> Option<Handle<SubFiniteElementSpace>> {
        Some(self.fespace.clone())
    }

    fn build_stiffness_blocks(
        &self,
        material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        let mat = material.expect("HeatConduction requires a material field");
        let mut block = SubMatrix::new(
            self.support.clone(),
            self.support.clone(),
            vec![DUAL_VAR.to_string()],
            vec![PRIMAL_VAR.to_string()],
            DofOrdering::NodesThenVars,
            true,
        )?;
        assemble_stiffness(&self.fespace, mat, &mut block)?;
        Ok(vec![block])
    }

    fn label(&self) -> &'static str {
        "HeatConduction"
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let n = with(&self.support, |s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<HeatConduction>\n  primal var(s): {primal}\n  \
             dual var(s):   {dual}\n  support: {n} node(s)"
        )
    }
}

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
