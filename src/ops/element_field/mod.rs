//! Operators that **produce** an
//! [`ElementField`](crate::containers::element_field::ElementField) — the
//! per-Gauss-point quantities of the assembly chain.
//!
//! - *kinematics* — [`gradient`](fn@gradient) (`∇f` of a nodal field),
//!   [`deformation`](fn@deformation) (the strain of a displacement field,
//!   linearized `ε` or finite, per [`StrainMeasure`]) and its structural variants
//!   [`beam_deformation`](fn@beam_deformation),
//!   [`shell_deformation`](fn@shell_deformation),
//!   [`interp_to_gauss`](fn@interp_to_gauss),
//!   [`thermal_strain`](fn@thermal_strain);
//! - *elastic operator* — [`elastic_stress`](fn@elastic_stress) `σ = D:ε` and
//!   its inverse [`elastic_strain`](fn@elastic_strain) (Cast3M `ELAS`);
//! - *material data* — [`material_field`](fn@material_field) & co., the
//!   per-zone coefficients the assemblers read;
//! - *constitutive law* — [`behavior::integrate`](fn@behavior::integrate)
//!   (Cast3M `COMP`), the exact, possibly non-linear counterpart of the
//!   [`crate::ops::matrix::stiffness`] linearization;
//! - [`cell_integral`](fn@cell_integral), one integral per cell (Cast3M
//!   `INTG 'ELEM'`);
//! - [`consolidate`](fn@consolidate), the aggregate-level fusion.

pub mod beam_deformation;
pub mod behavior;
pub mod cell_integral;
pub mod consolidate;
pub mod deformation;
pub mod elastic;
pub mod gradient;
pub mod interp_to_gauss;
pub mod mask;
pub mod material_field;
pub mod methods;
pub mod shell_deformation;
pub mod thermal_strain;

pub use beam_deformation::beam_deformation;
pub use cell_integral::cell_integral;
pub use consolidate::{check_unique_component_per_support, consolidate};
pub use deformation::{deformation, StrainMeasure};
pub use elastic::{elastic_strain, elastic_stress};
pub use gradient::gradient;
pub use interp_to_gauss::interp_to_gauss;
pub use mask::{mask, mask_sub};
pub use material_field::{material_field, material_field_per_sub_model, sub_material_field};
pub use shell_deformation::shell_deformation;
pub use thermal_strain::thermal_strain;
