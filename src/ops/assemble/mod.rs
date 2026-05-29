//! Assembly operators — turn a [`crate::containers::model::Model`] into a
//! [`crate::containers::matrix::Matrix`] (stiffness, mass) or
//! [`crate::containers::node_field::NodeField`] (RHS).
//!
//! The per-physics integrands live in [`crate::models`]
//! (`heat_conduction`, `dirichlet`, …). This layer orchestrates the
//! loop over sub-models, the DOF layout, and boundary-condition
//! application.

use crate::aggregate::Aggregate;
use crate::containers::element_field::SubElementField;
use crate::containers::matrix::Matrix;
use crate::containers::model::Model;
use crate::error::Result;
use crate::store::{insert, with, Handle};

/// Assemble the stiffness matrix `K` for `model`.
///
/// `material` is a per-element field carrying the material data required
/// by each physics (e.g. conductivity `"k"` for `HeatConduction`).
/// It is passed at call-site rather than stored in the model, keeping
/// the model immutable and decoupled from transient material state.
///
/// Each [`crate::containers::model::SubModel`] contributes one or more
/// [`crate::containers::matrix::SubMatrix`] blocks
/// (`HeatConduction` → 1 block, `Dirichlet` → C + Cᵀ).
/// The aggregate is finalized before being returned.
pub fn stiffness(model: &Model, material: &Handle<SubElementField>) -> Result<Matrix> {
    let mut k = Matrix::empty();
    for h in model {
        let blocks = with(h, |sub| sub.build_stiffness_blocks(Some(material)))??;
        for block in blocks {
            k.add_sub(insert(block))?;
        }
    }
    k.finalize()?;
    Ok(k)
}
