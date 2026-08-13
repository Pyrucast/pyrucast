//! Méthodes de délégation — la face « sujet » des opérateurs de ce module.
//!
//! Voir `CONVENTIONS.md` § « Le verbe exposé aussi en méthode ». La fonction
//! libre reste la forme canonique ; ces méthodes ne contiennent aucune logique.
//!
//! Ne sont **pas** exposés ici, faute de sens pour toute instance du type :
//! `deformation`, `beam_deformation` (elles exigent des
//! composantes de déplacement `u_x`/`u_y`/`u_z`) et `thermal_strain` (elle
//! exige une température, et un `alpha` dans le matériau). Elles restent des
//! fonctions libres seules.
//!
//! `sub_material_field` devient `SubModel::material_field` : le type fournit
//! déjà le qualificatif `sub`, le nom de la méthode n'a pas à le porter.

use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::model::{Model, SubModel};
use crate::containers::node_field::NodeField;
use crate::error::Result;

impl ElementField {
    /// Voir [`element_field::consolidate`](fn@crate::ops::element_field::consolidate).
    pub fn consolidate(&self) -> Result<ElementField> {
        crate::ops::element_field::consolidate(self)
    }
}

impl NodeField {
    /// Voir [`element_field::gradient`](fn@crate::ops::element_field::gradient).
    pub fn gradient(&self, fespace: &FiniteElementSpace) -> Result<ElementField> {
        crate::ops::element_field::gradient(self, fespace)
    }

    /// Voir [`element_field::interp_to_gauss`](fn@crate::ops::element_field::interp_to_gauss).
    pub fn interp_to_gauss(&self, fespace: &FiniteElementSpace) -> Result<ElementField> {
        crate::ops::element_field::interp_to_gauss(self, fespace)
    }
}

impl Model {
    /// Voir [`element_field::material_field`](fn@crate::ops::element_field::material_field).
    pub fn material_field(&self, components_and_values: &[(&str, f64)]) -> Result<ElementField> {
        crate::ops::element_field::material_field(self, components_and_values)
    }

    /// Voir [`element_field::material_field_per_sub_model`](fn@crate::ops::element_field::material_field_per_sub_model).
    pub fn material_field_per_sub_model(
        &self,
        per_sub_model: &[&[(&str, f64)]],
    ) -> Result<ElementField> {
        crate::ops::element_field::material_field_per_sub_model(self, per_sub_model)
    }

    /// Voir [`element_field::behavior::integrate`](fn@crate::ops::element_field::behavior::integrate).
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
    /// Voir [`element_field::sub_material_field`](fn@crate::ops::element_field::sub_material_field).
    pub fn material_field(&self, components_and_values: &[(&str, f64)]) -> Result<SubElementField> {
        crate::ops::element_field::sub_material_field(self, components_and_values)
    }
}
