//! Field operators — derivations and transformations on
//! [`crate::containers::node_field::SubNodeField`] / [`crate::containers::element_field::ElementField`].
//!
//! [`gradient`](fn@gradient) (`∇f` of a nodal field) and [`deformation`](fn@deformation) (the linearized
//! strain `ε` of a displacement field) are the purely geometric producers of
//! the per-element field that [`crate::ops::behavior::integrate`] consumes;
//! [`interp_to_gauss`](fn@interp_to_gauss) interpolates a nodal field's values to the Gauss points.
//! More to come: `project_to_nodes`.
//! Component-wise scalar maths and per-element operations stay close to the
//! data (on the field types) — this module is for operations that **cross
//! containers** (mesh + field, fe_space + field).

pub mod band;
pub mod beam_deformation;
pub mod consolidate_element;
pub mod consolidate_node;
pub mod coordinates;
pub mod deformation;
pub mod divergence;
pub mod elementwise;
pub mod frame_deformation;
pub mod gradient;
pub mod integral;
pub mod interp_to_gauss;
pub mod mask;
pub mod merge;
pub mod restrict;
pub mod select;
pub mod thermal_strain;

pub use band::Band;
pub use beam_deformation::beam_deformation;
pub use consolidate_element::{check_unique_component_per_support, consolidate_element};
pub use consolidate_node::consolidate_node;
pub use coordinates::{coordinates, displace, set_coordinates};
pub use deformation::deformation;
pub use divergence::divergence;
pub use elementwise::{abs, cos, cosh, exp, log, log10, sin, sinh, sqrt, tan, tanh};
pub use frame_deformation::frame_deformation;
pub use gradient::gradient;
pub use integral::{integral, integral_element};
pub use interp_to_gauss::interp_to_gauss;
pub use mask::{mask_cells, mask_nodes, mask_sub_cells, mask_sub_nodes};
pub use merge::merge;
pub use restrict::{restrict, restrict_like};
pub use select::{select_cells, select_nodes, select_sub_cells, select_sub_nodes};
pub use thermal_strain::thermal_strain;
