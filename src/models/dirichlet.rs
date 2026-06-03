//! Dirichlet constraint via Lagrange multipliers.
//!
//! Construction creates one **multiplier node** per constrained node (at
//! the same coordinates), owned by a dedicated POI1 [`SubMesh`] whose
//! `Drop` releases the multipliers when the sub-model dies. Assembly
//! writes the `C` / `Cᵀ` Lagrange block into the global matrix —
//! enforces `u_n = u_d_n` once the user fills the load `NodeField` at
//! `(multiplier_node, primal_var)`.

use crate::containers::element_field::SubElementField;
use crate::containers::matrix::{DofOrdering, SubMatrix};
use crate::containers::mesh::{ElementType, Node, NodeId};
use crate::containers::mesh::SubMesh;
use crate::dump::DumpOptions;
use crate::error::{PyrucastError, Result};
use crate::models::Physics;
use crate::store::{insert, with, with_mut, Handle};
use serde::{Deserialize, Serialize};

/// Multiplier-node name auto-generated from a primal variable name.
pub fn multiplier_name(primal_var: &str) -> String {
    format!("lambda_{primal_var}")
}

/// Materials needed to instantiate the Dirichlet variant.
///
/// [`Dirichlet::new`] returns this bundle's fields moved into the struct.
/// The node sequences are recoverable at any time by reading the POI1
/// supports' connectivity (size immutable on insert into the store).
pub struct Built {
    pub constrained_support: Handle<SubMesh>,
    pub multiplier_support: Handle<SubMesh>,
}

/// Build the Lagrange-multiplier infrastructure for a Dirichlet constraint
/// on `constrained_nodes`. The Configuration is taken from the nodes
/// themselves (every [`Node`] carries its own); one new multiplier node
/// per constrained node is added to it at the same coordinates.
pub fn build(constrained_nodes: &[Node]) -> Result<Built> {
    let config = constrained_nodes
        .first()
        .ok_or_else(|| {
            PyrucastError::Message("Dirichlet: constrained_nodes must not be empty".into())
        })?
        .configuration();

    // POI1 SubMesh that owns the per-node refcounts on the constrained
    // nodes. `poi1_from_nodes` increfs each (and rolls back on failure).
    let constrained_support = insert(SubMesh::poi1_from_nodes(constrained_nodes)?);

    // Create the multiplier nodes at the same coordinates as the
    // constrained ones, then hand each multiplier's initial refcount
    // (left by `add_node`) over to a POI1 SubMesh via `add_cell_taking`
    // — ownership transfer, no extra incref/decref.
    let mut coords: Vec<Vec<f64>> = Vec::with_capacity(constrained_nodes.len());
    with(&config, |c| -> Result<()> {
        for n in constrained_nodes {
            coords.push(c.coord(n.id())?.to_vec());
        }
        Ok(())
    })??;

    let multiplier_nodes: Vec<NodeId> = with_mut(&config, |c| -> Result<Vec<NodeId>> {
        let mut out = Vec::with_capacity(coords.len());
        for coord in &coords {
            out.push(c.add_node(coord)?);
        }
        Ok(out)
    })??;

    let mut multiplier_sm = SubMesh::new(config, ElementType::POI1);
    for &nid in &multiplier_nodes {
        multiplier_sm.add_cell_taking(&[nid])?;
    }
    let multiplier_support = insert(multiplier_sm);

    Ok(Built {
        constrained_support,
        multiplier_support,
    })
}

/// Dirichlet constraint imposed via Lagrange multipliers.
///
/// For each constrained primary node `n`, the system is augmented with one
/// multiplier `λ_n` (a new node introduced on the fly at the same
/// coordinates as `n`) and a pair of unit entries enforcing `u_n = u_d_n`.
/// The imposed value `u_d_n` is **not** stored here: the user supplies it
/// through the load `NodeField` at the multiplier node's `<var>` component.
///
/// - primal variable: `"lambda_<primal_var>"` (multiplier DOFs, on
///   `multiplier_support`).
/// - dual variable:   `<primal_var>` itself (constraint equation rows).
/// - `primal_dual` is the dual variable name of the **primary physics**
///   this constraint targets (`"q"` for heat conduction, `"f_x"` for
///   elasticity in `x`, …): it tells the constraint where in the row index
///   to write the `Cᵀ` block.
#[derive(Clone, Serialize, Deserialize)]
pub struct Dirichlet {
    pub(crate) primal_var: String,
    pub(crate) primal_dual: String,
    /// POI1 SubMesh holding the per-node refcounts on the constrained
    /// nodes. Connectivity is the constrained-node sequence (immutable
    /// once inserted into the store).
    pub(crate) constrained_support: Handle<SubMesh>,
    /// POI1 SubMesh owning the multiplier nodes (one cell per multiplier,
    /// same order as the constrained nodes). Holds the only refcount on
    /// each multiplier; its `Drop` collects them.
    pub(crate) multiplier_support: Handle<SubMesh>,
}

impl Dirichlet {
    /// Build a Dirichlet constraint on `constrained_nodes`. `primal_var` is
    /// the constrained primary variable; `primal_dual` is the dual variable
    /// name of the primary physics it targets (see the struct docs). One
    /// new multiplier node per constrained node is created in the
    /// `Configuration`.
    pub fn new(
        primal_var: String,
        primal_dual: String,
        constrained_nodes: &[Node],
    ) -> Result<Self> {
        let built = build(constrained_nodes)?;
        Ok(Self {
            primal_var,
            primal_dual,
            constrained_support: built.constrained_support,
            multiplier_support: built.multiplier_support,
        })
    }
}

impl Physics for Dirichlet {
    fn primal_vars(&self) -> Vec<String> {
        vec![multiplier_name(&self.primal_var)]
    }

    fn dual_vars(&self) -> Vec<String> {
        vec![self.primal_var.clone()]
    }

    fn multiplier_support(&self) -> Option<&Handle<SubMesh>> {
        Some(&self.multiplier_support)
    }

    fn build_stiffness_blocks(
        &self,
        _material: Option<&Handle<SubElementField>>,
    ) -> Result<Vec<SubMatrix>> {
        let constrained_nodes: Vec<NodeId> =
            with(&self.constrained_support, |s| s.connectivity().to_vec())?;
        let multiplier_nodes: Vec<NodeId> =
            with(&self.multiplier_support, |s| s.connectivity().to_vec())?;
        let lambda_name = multiplier_name(&self.primal_var);
        // C block: rows = multiplier × primal_var, cols = constrained × primal_var
        let mut c_block = SubMatrix::new(
            self.multiplier_support.clone(),
            self.constrained_support.clone(),
            vec![self.primal_var.clone()],
            vec![self.primal_var.clone()],
            DofOrdering::NodesThenVars,
            true,
        )?;
        // Cᵀ block: rows = constrained × primal_dual, cols = multiplier × lambda
        let mut ct_block = SubMatrix::new(
            self.constrained_support.clone(),
            self.multiplier_support.clone(),
            vec![self.primal_dual.clone()],
            vec![lambda_name],
            DofOrdering::NodesThenVars,
            true,
        )?;
        assemble_blocks(
            &constrained_nodes,
            &multiplier_nodes,
            &self.primal_var,
            &self.primal_dual,
            &mut c_block,
            &mut ct_block,
        )?;
        Ok(vec![c_block, ct_block])
    }

    fn label(&self) -> &'static str {
        "Dirichlet"
    }

    fn display(&self) -> String {
        let n = with(&self.constrained_support, |s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Dirichlet({})>: {} constrained node(s)",
            self.primal_var, n
        )
    }

    fn render(&self, _opts: &DumpOptions) -> String {
        let primal = self.primal_vars().join(", ");
        let dual = self.dual_vars().join(", ");
        let nc = with(&self.constrained_support, |s| s.cell_count()).unwrap_or(0);
        let nm = with(&self.multiplier_support, |s| s.cell_count()).unwrap_or(0);
        format!(
            "SubModel<Dirichlet({primal_var})>\n  primal var(s): {primal} (multipliers)\n  \
             dual var(s):   {dual}\n  targets primary dual: {primal_dual}\n  \
             constrained: {nc} node(s)\n  multipliers: {nm} node(s)",
            primal_var = self.primal_var,
            primal_dual = self.primal_dual,
        )
    }
}

/// Fill the C and Cᵀ blocks of a Dirichlet sub-model.
///
/// For each `(constrained_node, multiplier_node)` pair:
/// - **C entry**  into `c_block`  at `(multiplier_node, primal_var)` ×
///   `(constrained_node, primal_var)` = `1`;
/// - **Cᵀ entry** into `ct_block` at `(constrained_node, primal_dual)` ×
///   `(multiplier_node, lambda_<primal_var>)` = `1`.
pub fn assemble_blocks(
    constrained_nodes: &[NodeId],
    multiplier_nodes: &[NodeId],
    primal_var: &str,
    primal_dual: &str,
    c_block: &mut SubMatrix,
    ct_block: &mut SubMatrix,
) -> crate::error::Result<()> {
    let lambda_name = multiplier_name(primal_var);
    for (c_node, m_node) in constrained_nodes.iter().zip(multiplier_nodes.iter()) {
        c_block.add_entry(*m_node, primal_var, *c_node, primal_var, 1.0)?;
        ct_block.add_entry(*c_node, primal_dual, *m_node, &lambda_name, 1.0)?;
    }
    Ok(())
}
