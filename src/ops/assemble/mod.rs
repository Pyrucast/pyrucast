//! Assembly operators — turn a [`crate::containers::model::Model`] into a
//! [`crate::containers::matrix::Matrix`] (stiffness, mass) or
//! [`crate::containers::node_field::SubNodeField`] (RHS).
//!
//! The per-physics integrands live in [`crate::models`]
//! (`heat_conduction`, `dirichlet`, …). This layer orchestrates the
//! loop over sub-models, the DOF layout, and boundary-condition
//! application.
//!
//! # Material lookup
//!
//! Material data is supplied as an [`ElementField`] aggregate. For every
//! sub-model that needs material values (e.g. `HeatConduction`), the
//! assembler picks the [`SubElementField`] whose `SubFiniteElementSpace`
//! handle matches the sub-model's own FE subspace. This lets each zone
//! carry its own material — different conductivities, different
//! materials — without coupling the (declarative) model to the
//! (per-iteration, mutable) material state.
//!
//! Sub-models that don't need material data (`Dirichlet`, …) are
//! independent of the supplied `ElementField`; an `ElementField`
//! covering only some of the FE subspaces is therefore valid as long as
//! every material-hungry sub-model finds its match.

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::SubField;
use crate::containers::matrix::Matrix;
use crate::containers::model::Model;
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read, Handle};

pub mod flux;
pub use flux::{flux, FluxDensity};

/// Assemble the stiffness matrix `K` for `model`.
///
/// `materials` is an [`ElementField`] aggregate; each sub-model that
/// needs material data picks the [`SubElementField`] whose FE subspace
/// matches its own (zone-wise materials).
///
/// Each [`crate::containers::model::SubModel`] contributes one or more
/// [`crate::containers::matrix::SubMatrix`] blocks
/// (`HeatConduction` → 1 block, `Dirichlet` → C + Cᵀ).
/// The aggregate is finalized before being returned.
pub fn stiffness(model: &Model, materials: &ElementField) -> Result<Matrix> {
    let mut k = Matrix::empty();
    for sub_h in model {
        let blocks = {
            let sub = read(sub_h)?;
            // Generic over the physics: a sub-model needs material data iff
            // it declares a material FE subspace. No per-variant match.
            let material = match sub.material_fespace() {
                Some(fespace) => {
                    let m = materials.sub_for_fespace(&fespace)?;
                    if let Some(required) = sub.material_components() {
                        validate_material(&m, required)?;
                    }
                    Some(m)
                }
                None => None,
            };
            sub.build_stiffness_blocks(material.as_ref())?
        };
        for block in blocks {
            k.add_sub(insert(block))?;
        }
    }
    k.finalize()?;
    Ok(k)
}

/// Assemble the mass matrix `M` for `model`.
///
/// v0 stub: no physics has a mass term yet, so this returns an empty
/// finalized [`Matrix`] with the model's DOF layout. Kept alongside
/// [`stiffness`] so the assembler family lives in one place.
pub fn mass(model: &Model) -> Result<Matrix> {
    // `model` is unused until a physics introduces a mass term; binding it
    // documents the intended signature (assemble over the model's DOFs).
    let _ = model;
    let mut m = Matrix::empty();
    m.finalize()?;
    Ok(m)
}

/// Ensure `material` carries every component declared as required by
/// the physics. Errors with both lists for a clear message.
fn validate_material(material: &Handle<SubElementField>, required: &[&str]) -> Result<()> {
    let have: Vec<String> = read(material)?.components().to_vec();
    for req in required {
        if !have.iter().any(|c| c == req) {
            return Err(PyrucastError::Message(format!(
                "assemble::stiffness: required material component '{}' missing on \
                 SubElementField (has: [{}])",
                req,
                have.join(", ")
            )));
        }
    }
    Ok(())
}
