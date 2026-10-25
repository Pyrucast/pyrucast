//! Delegation methods — the "subject" face of this module's operators.
//!
//! See `CONVENTIONS.md` § "The verb also exposed as a method". The free
//! function stays the canonical form; these methods contain no logic.
//!
//! **Not** exposed here, for want of meaning for every instance of the type:
//! `deformation`, `beam_deformation` (they require displacement components
//! `u_x`/`u_y`/`u_z`) and `thermal_strain` (it requires a temperature, and an
//! `alpha` in the material). They stay
//! free functions only.
//!
//! `sub_material_field` becomes `SubModel::material_field`: the type supplies
//! the `sub` qualifier already, the method's name need not carry it.

use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::containers::node_field::NodeField;
use crate::error::Result;

impl ElementField {
    /// See [`element_field::consolidate`](fn@crate::ops::element_field::consolidate).
    pub fn consolidate(&self) -> Result<ElementField> {
        crate::ops::element_field::consolidate(self)
    }
}

impl NodeField {
    /// See [`element_field::gradient`](fn@crate::ops::element_field::gradient).
    pub fn gradient(&self, fespace: &FiniteElementSpace) -> Result<ElementField> {
        crate::ops::element_field::gradient(self, fespace)
    }

    /// See [`element_field::interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss).
    pub fn interp_to_gauss(&self, fespace: &FiniteElementSpace) -> Result<ElementField> {
        crate::ops::element_field::interp_to_gauss(self, fespace)
    }
}

impl Model {
    /// See [`element_field::material_field`](fn@crate::ops::element_field::material_field).
    pub fn material_field(&self, components_and_values: &[(&str, f64)]) -> Result<ElementField> {
        crate::ops::element_field::material_field(self, components_and_values)
    }

    /// See [`element_field::material_field_per_sub_model`](fn@crate::ops::element_field::material_field_per_sub_model).
    pub fn material_field_per_sub_model(
        &self,
        per_sub_model: &[&[(&str, f64)]],
    ) -> Result<ElementField> {
        crate::ops::element_field::material_field_per_sub_model(self, per_sub_model)
    }

    /// See [`element_field::behavior::integrate`](fn@crate::ops::element_field::behavior::integrate).
    pub fn integrate_behavior(
        &self,
        deformation: &ElementField,
        prev: Option<&ElementField>,
        materials: &ElementField,
        dt: Option<f64>,
    ) -> Result<ElementField> {
        crate::ops::element_field::behavior::integrate(self, deformation, prev, materials, dt)
    }
}

impl SubModel {
    /// See [`element_field::sub_material_field`](fn@crate::ops::element_field::sub_material_field).
    pub fn material_field(&self, components_and_values: &[(&str, f64)]) -> Result<SubElementField> {
        crate::ops::element_field::sub_material_field(self, components_and_values)
    }
}
