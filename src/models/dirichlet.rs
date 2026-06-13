//! Dirichlet constraint via Lagrange multipliers.
//!
//! A Dirichlet sub-model imposes `u = u_d` on a set of constrained nodes. It
//! is a *constraint*, not a volumetric physics: it carries no material and no
//! constitutive law. It is built from **two meshes supplied by the user** — it
//! creates no node and never mutates the
//! [`Configuration`](crate::containers::mesh::Configuration):
//!
//! - `imposed_mesh`  — POI1 (for now): the constrained nodes (shared with the
//!   target physics), one node per cell;
//! - `multiplier_mesh` — POI1: the support of the Lagrange multipliers, paired
//!   element-for-element with `imposed_mesh` (same submesh structure, same
//!   per-submesh cell count). Typically built from `imposed_mesh` with the
//!   generic [`barycenter`](crate::ops::mesher::barycenter()) mesher (colocated fresh nodes),
//!   but the user is free to colocate, offset, or even reuse the constrained
//!   nodes themselves.
//!
//! # Variables
//!
//! A Dirichlet sub-model imposes the **primal of another model** yet owns its
//! own pair of variables; the names keep the two apart:
//!
//! - `imposed_variable` (e.g. `"T"`) — the constrained primal of the **target**
//!   physics (a *column* of the target's stiffness);
//! - `target_dual` (e.g. `"q"`) — the **target's** dual variable; the row into
//!   which the constraint reaction `Cᵀ` is added (it cannot be derived from
//!   `imposed_variable` — the `T → q` map is target-specific);
//! - `multiplier` (default `lambda_<imposed_variable>`) — this sub-model's own
//!   **primal**: the Lagrange multiplier, an unknown of the augmented system,
//!   whose solved value is the **reaction**;
//! - `imposed_value` (default `imposed_<imposed_variable>`) — this sub-model's
//!   own **dual**: the constraint-equation row, and the **slot** at which the
//!   user writes the imposed value `u_d` in the load `SubNodeField` (on the
//!   multiplier node).
//!
//! `multiplier` and `imposed_value` are derived from `imposed_variable` but can
//! be overridden by hand.

use crate::aggregate::Aggregate;
use crate::containers::element_field::SubElementField;
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{ElementType, Mesh, NodeId};
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::Physics;
use crate::store::read;
use serde::{Deserialize, Serialize};

/// Default multiplier (primal) name for a constrained variable: `lambda_<v>`.
pub fn default_multiplier(imposed_variable: &str) -> String {
    format!("lambda_{imposed_variable}")
}

/// Default imposed-value (dual) name for a constrained variable: `imposed_<v>`.
pub fn default_imposed_value(imposed_variable: &str) -> String {
    format!("imposed_{imposed_variable}")
}

/// Dirichlet constraint imposed via Lagrange multipliers.
///
/// See the module documentation for the meaning of the four variable names and
/// the two meshes. The imposed value `u_d` is **not** stored here: the user
/// supplies it through the load `SubNodeField` at the multiplier node's
/// `imposed_value` component.
#[derive(Serialize, Deserialize)]
pub struct Dirichlet {
    /// Constrained primal of the target physics (e.g. `"T"`).
    pub(crate) imposed_variable: String,
    /// Target physics's dual; row where the reaction `Cᵀ` lands (e.g. `"q"`).
    pub(crate) target_dual: String,
    /// This sub-model's primal — the Lagrange multiplier (e.g. `"lambda_T"`).
    pub(crate) multiplier: String,
    /// This sub-model's dual — constraint row + imposed-value slot (e.g. `"imposed_T"`).
    pub(crate) imposed_value: String,
    /// POI1 mesh of the constrained nodes (one node per cell), as given by the
    /// user. Shares the submeshes (refcounts keep the nodes alive).
    pub(crate) imposed_mesh: Mesh,
    /// POI1 mesh of the multiplier nodes, paired element-for-element with
    /// `imposed_mesh` (same submesh structure, same per-submesh cell count).
    pub(crate) multiplier_mesh: Mesh,
}

impl Dirichlet {
    /// Build a Dirichlet constraint imposing `imposed_variable = u_d` on the
    /// nodes of `imposed_mesh`, with multipliers living on `multiplier_mesh`.
    ///
    /// `target_dual` is the dual variable of the target physics (e.g. `"q"`).
    /// `multiplier` / `imposed_value` default to `lambda_<imposed_variable>` /
    /// `imposed_<imposed_variable>` when `None`. Both meshes must be POI1 (for
    /// now), share one [`Configuration`](crate::containers::mesh::Configuration),
    /// and pair element-for-element (same number of submeshes, same cell count
    /// per submesh).
    pub fn new(
        imposed_variable: String,
        target_dual: String,
        imposed_mesh: &Mesh,
        multiplier_mesh: &Mesh,
        multiplier: Option<String>,
        imposed_value: Option<String>,
    ) -> Result<Self> {
        if imposed_mesh.cell_count()? == 0 {
            return Err(PyrucastError::Message(
                "Dirichlet: imposed_mesh must constrain at least one node".into(),
            ));
        }
        let n_sub = imposed_mesh.len();
        if multiplier_mesh.len() != n_sub {
            return Err(PyrucastError::Message(format!(
                "Dirichlet: imposed_mesh has {} submesh(es) but multiplier_mesh has {}",
                n_sub,
                multiplier_mesh.len()
            )));
        }
        // NodeIds are Configuration-relative: both meshes must share it.
        let cfg_i = imposed_mesh.configuration()?;
        let cfg_m = multiplier_mesh.configuration()?;
        if cfg_i.index() != cfg_m.index() || cfg_i.generation() != cfg_m.generation() {
            return Err(PyrucastError::Message(
                "Dirichlet: imposed_mesh and multiplier_mesh must share a Configuration".into(),
            ));
        }
        // POI1 (for now) + element-for-element pairing (equal cell counts).
        for i in 0..n_sub {
            let imp = imposed_mesh.get(i)?;
            let mult = multiplier_mesh.get(i)?;
            let (iet, icount) = {
                let s = read(&imp)?;
                (s.element_type(), s.cell_count())
            };
            let (met, mcount) = {
                let s = read(&mult)?;
                (s.element_type(), s.cell_count())
            };
            if iet != ElementType::POI1 {
                return Err(PyrucastError::Message(format!(
                    "Dirichlet: imposed_mesh submesh {i} must be POI1, got {iet}"
                )));
            }
            if met != ElementType::POI1 {
                return Err(PyrucastError::Message(format!(
                    "Dirichlet: multiplier_mesh submesh {i} must be POI1, got {met}"
                )));
            }
            if icount != mcount {
                return Err(PyrucastError::Message(format!(
                    "Dirichlet: submesh {i} has {icount} constrained node(s) but \
                     {mcount} multiplier(s)"
                )));
            }
        }

        let multiplier = multiplier.unwrap_or_else(|| default_multiplier(&imposed_variable));
        let imposed_value =
            imposed_value.unwrap_or_else(|| default_imposed_value(&imposed_variable));

        Ok(Self {
            imposed_variable,
            target_dual,
            multiplier,
            imposed_value,
            imposed_mesh: share(imposed_mesh)?,
            multiplier_mesh: share(multiplier_mesh)?,
        })
    }
}

impl Physics for Dirichlet {
    fn primal_vars(&self) -> Vec<String> {
        vec![self.multiplier.clone()]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![self.imposed_value.clone()]
    }

    fn multiplier_mesh(&self) -> Option<&Mesh> {
        Some(&self.multiplier_mesh)
    }

    fn build_stiffness_blocks(
        &self,
        _material: Option<&crate::store::Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        // One C / Cᵀ pair per submesh pair — the user's submeshes are used
        // directly as row/col supports (no flattening), so the submesh
        // structure carried through `barycenter` is preserved.
        let mut blocks = Vec::with_capacity(self.imposed_mesh.len() * 2);
        for i in 0..self.imposed_mesh.len() {
            let imp_sm = self.imposed_mesh.get(i)?;
            let mult_sm = self.multiplier_mesh.get(i)?;
            let imposed_nodes: Vec<NodeId> = read(&imp_sm)?.connectivity().to_vec();
            let multiplier_nodes: Vec<NodeId> = read(&mult_sm)?.connectivity().to_vec();

            // C block: rows = multiplier × imposed_value,
            //          cols = imposed × imposed_variable.
            let mut c_block = SubMatrix::new(
                mult_sm.clone(),
                imp_sm.clone(),
                vec![self.imposed_value.clone()],
                vec![self.imposed_variable.clone()],
                DofOrdering::NodesThenVars,
                false,
            )?;
            // Cᵀ block: rows = imposed × target_dual, cols = multiplier × multiplier.
            let mut ct_block = SubMatrix::new(
                imp_sm.clone(),
                mult_sm.clone(),
                vec![self.target_dual.clone()],
                vec![self.multiplier.clone()],
                DofOrdering::NodesThenVars,
                false,
            )?;
            assemble_blocks(
                &imposed_nodes,
                &multiplier_nodes,
                &self.imposed_variable,
                &self.target_dual,
                &self.multiplier,
                &self.imposed_value,
                &mut c_block,
                &mut ct_block,
            )?;
            blocks.push(c_block);
            blocks.push(ct_block);
        }
        Ok(blocks)
    }

    fn label(&self) -> &'static str {
        "Dirichlet"
    }

    fn display(&self) -> String {
        let n = self.imposed_mesh.cell_count().unwrap_or(0);
        format!(
            "SubModel<Dirichlet({})>: {} constrained node(s)",
            self.imposed_variable, n
        )
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let nc = self.imposed_mesh.cell_count().unwrap_or(0);
        let nm = self.multiplier_mesh.cell_count().unwrap_or(0);
        format!(
            "SubModel<Dirichlet({imposed_variable})>\n  primal var(s): {multiplier} \
             (multiplier)\n  dual var(s):   {imposed_value} (imposed value)\n  \
             targets primary dual: {target_dual}\n  constrained: {nc} node(s)\n  \
             multipliers: {nm} node(s)",
            imposed_variable = self.imposed_variable,
            multiplier = self.multiplier,
            imposed_value = self.imposed_value,
            target_dual = self.target_dual,
        )
    }
}

/// Clone a mesh by sharing its submeshes — increfs each submesh handle so the
/// nodes stay alive for the lifetime of the sub-model.
fn share(mesh: &Mesh) -> Result<Mesh> {
    let mut out = Mesh::empty();
    for sm in mesh {
        out.add_sub(sm.clone())?;
    }
    Ok(out)
}

/// Fill the C and Cᵀ blocks of a Dirichlet sub-model for one submesh pair.
///
/// For each `(imposed_node, multiplier_node)` pair (element-for-element):
/// - **C entry**  into `c_block`  at `(multiplier_node, imposed_value)` ×
///   `(imposed_node, imposed_variable)` = `1` — the constraint equation;
/// - **Cᵀ entry** into `ct_block` at `(imposed_node, target_dual)` ×
///   `(multiplier_node, multiplier)` = `1` — the reaction in the target's
///   dual row.
#[allow(clippy::too_many_arguments)]
pub fn assemble_blocks(
    imposed_nodes: &[NodeId],
    multiplier_nodes: &[NodeId],
    imposed_variable: &str,
    target_dual: &str,
    multiplier: &str,
    imposed_value: &str,
    c_block: &mut SubMatrix,
    ct_block: &mut SubMatrix,
) -> Result<()> {
    for (imp, mult) in imposed_nodes.iter().zip(multiplier_nodes.iter()) {
        c_block.add_entry(*mult, imposed_value, *imp, imposed_variable, 1.0)?;
        ct_block.add_entry(*imp, target_dual, *mult, multiplier, 1.0)?;
    }
    Ok(())
}
