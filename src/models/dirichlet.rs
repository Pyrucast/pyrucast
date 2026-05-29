//! Dirichlet constraint via Lagrange multipliers.
//!
//! Construction creates one **multiplier node** per constrained node (at
//! the same coordinates), owned by a dedicated POI1 [`SubMesh`] whose
//! `Drop` releases the multipliers when the sub-model dies. Assembly
//! writes the `C` / `Cᵀ` Lagrange block into the global matrix —
//! enforces `u_n = u_d_n` once the user fills the load `NodeField` at
//! `(multiplier_node, primal_var)`.

use crate::containers::mesh::configuration::{Configuration, NodeId};
use crate::containers::mesh::element_type::ElementType;
use crate::error::{PyrucastError, Result};
use crate::containers::matrix::SubMatrix;
use crate::containers::mesh::SubMesh;
use crate::store::{insert, with, with_mut, Handle};

/// Multiplier-node name auto-generated from a primal variable name.
pub fn multiplier_name(primal_var: &str) -> String {
    format!("lambda_{primal_var}")
}

/// Materials needed to instantiate the Dirichlet variant.
///
/// `build` returns this bundle and the [`crate::containers::model::Physics::Dirichlet`]
/// variant simply moves its fields in. The node sequences are recoverable
/// at any time by reading the POI1 supports' connectivity (size immutable
/// on insert into the store).
pub struct Built {
    pub constrained_support: Handle<SubMesh>,
    pub multiplier_support: Handle<SubMesh>,
}

/// Build the Lagrange-multiplier infrastructure for a Dirichlet constraint
/// on `constrained_nodes` inside `config`. One new multiplier node per
/// constrained node is added to `config` at the same coordinates.
pub fn build(
    config: Handle<Configuration>,
    constrained_nodes: &[NodeId],
) -> Result<Built> {
    if constrained_nodes.is_empty() {
        return Err(PyrucastError::Message(
            "Dirichlet: constrained_nodes must not be empty".into(),
        ));
    }

    // POI1 SubMesh that owns the per-node refcounts on the constrained
    // nodes. `add_cell` increfs each; if any fails, the partial SubMesh's
    // `Drop` rolls back via `?`.
    let mut constrained_sm = SubMesh::new(config.clone(), ElementType::POI1);
    for &nid in constrained_nodes {
        constrained_sm.add_cell(&[nid])?;
    }
    let constrained_support = insert(constrained_sm);

    // Create the multiplier nodes at the same coordinates as the
    // constrained ones, then hand each multiplier's initial refcount
    // (left by `add_node`) over to a POI1 SubMesh via `add_cell_taking`
    // — ownership transfer, no extra incref/decref.
    let mut coords: Vec<Vec<f64>> = Vec::with_capacity(constrained_nodes.len());
    with(&config, |c| -> Result<()> {
        for &nid in constrained_nodes {
            coords.push(c.coord(nid)?.to_vec());
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
