//! Assembly operators — turn a [`crate::containers::model::Model`] into a
//! [`crate::containers::matrix::Matrix`] (stiffness, mass) or
//! [`crate::containers::node_field::NodeField`] (RHS).
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
use crate::containers::finite_element_space::SubFiniteElementSpace;
use crate::containers::matrix::Matrix;
use crate::containers::model::{Model, Physics};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, with, Handle};

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
        let blocks = with(sub_h, |sub| -> Result<_> {
            let material = match sub.physics() {
                Physics::HeatConduction { fespace, .. } => {
                    let m = find_material_for_fespace(materials, fespace)?;
                    let required = sub.material_components().expect(
                        "HeatConduction always declares its material components",
                    );
                    validate_material(&m, required)?;
                    Some(m)
                }
                Physics::Dirichlet { .. } => None,
            };
            sub.build_stiffness_blocks(material.as_ref())
        })??;
        for block in blocks {
            k.add_sub(insert(block))?;
        }
    }
    k.finalize()?;
    Ok(k)
}

/// Find the [`SubElementField`] in `materials` whose FE subspace handle
/// matches `fespace`. Errors if no match is found.
fn find_material_for_fespace(
    materials: &ElementField,
    fespace: &Handle<SubFiniteElementSpace>,
) -> Result<Handle<SubElementField>> {
    for sub_h in materials {
        let matches = with(sub_h, |s| {
            let f = s.fespace();
            f.index() == fespace.index() && f.generation() == fespace.generation()
        })?;
        if matches {
            return Ok(sub_h.clone());
        }
    }
    Err(PyrucastError::Message(format!(
        "assemble::stiffness: no SubElementField in the materials aggregate matches \
         the SubFiniteElementSpace at slot {} (generation {})",
        fespace.index(),
        fespace.generation()
    )))
}

/// Ensure `material` carries every component declared as required by
/// the physics. Errors with both lists for a clear message.
fn validate_material(
    material: &Handle<SubElementField>,
    required: &[&str],
) -> Result<()> {
    let have: Vec<String> = with(material, |s| s.components().to_vec())?;
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
