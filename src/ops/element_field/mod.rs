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
//! - *the model's `B`* — [`strain`](fn@strain), the input of every law of a
//!   model, each from its own producer above (Cast3M `EPSI`);
//! - *elastic operator* — [`elastic_stress`](fn@elastic_stress) `σ = D:ε` and
//!   its inverse [`elastic_strain`](fn@elastic_strain) (Cast3M `ELAS`);
//! - *equivalent stress* — [`von_mises`](fn@von_mises), von Mises or Hill's
//!   anisotropic form;
//! - *tensor contractions* — [`tensor_dot`](fn@tensor_dot) `A·B` and
//!   [`tensor_ddot`](fn@tensor_ddot) `A:B`, at every Gauss point;
//! - *material data* — [`material_field`](fn@material_field) & co., the
//!   per-zone coefficients the assemblers read, and
//!   [`rotate_frame`](fn@rotate_frame), their axes turned with the part;
//! - *constitutive law* — [`behavior::integrate`](fn@behavior::integrate)
//!   (Cast3M `COMP`), the exact, possibly non-linear counterpart of the
//!   [`crate::ops::matrix::stiffness`] linearization;
//! - [`cell_integral`](fn@cell_integral), one integral per cell (Cast3M
//!   `INTG 'ELEM'`);
//! - [`consolidate`](fn@consolidate), the aggregate-level fusion;
//! - [`restrict`](fn@restrict) and [`restrict_like`](fn@restrict_like), a
//!   field read on other subspaces with the same cells and the same
//!   finite-element definition.

pub mod beam_deformation;
pub mod behavior;
pub mod cell_integral;
pub mod consolidate;
pub mod contraction;
pub mod deformation;
pub mod elastic;
pub(crate) mod fit;
pub mod gradient;
pub mod interp_to_gauss;
pub mod mask;
pub mod material_field;
pub mod methods;
pub mod restrict;
pub mod rotate_frame;
pub mod shell_deformation;
pub mod strain;
pub mod stress_measure;
pub mod thermal_strain;
pub mod von_mises;

pub use beam_deformation::beam_deformation;
pub use behavior::initial_state;
pub use cell_integral::cell_integral;
pub use consolidate::{check_unique_component_per_support, consolidate};
pub use contraction::{tensor_ddot, tensor_dot};
pub use deformation::{deformation, StrainMeasure};
pub use elastic::{elastic_strain, elastic_stress};
pub use gradient::gradient;
pub use interp_to_gauss::interp_to_gauss;
pub use mask::{mask, mask_sub};
pub use material_field::{material_field, material_field_per_sub_model, sub_material_field};
pub use restrict::{restrict, restrict_like};
pub use rotate_frame::rotate_frame;
pub use shell_deformation::shell_deformation;
pub use strain::strain;
pub use stress_measure::{stress_measure, StressMeasure};
pub use thermal_strain::thermal_strain;
pub use von_mises::{von_mises, VON_MISES};
