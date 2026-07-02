//! Field operators — derivations and transformations on
//! [`crate::containers::node_field::SubNodeField`] / [`crate::containers::element_field::ElementField`].
//!
//! [`gradient`](fn@gradient) (`∇f` of a nodal field) and [`deformation`](fn@deformation) (the linearized
//! strain `ε` of a displacement field) are the purely geometric producers of
//! the per-element field that [`crate::ops::behavior::integrate`] consumes.
//! More to come: `divergence`, `interp_to_gauss`, `project_to_nodes`.
//! Component-wise scalar maths and per-element operations stay close to the
//! data (on the field types) — this module is for operations that **cross
//! containers** (mesh + field, fe_space + field).

pub mod band;
pub mod beam_deformation;
pub mod consolidate;
pub mod consolidate_element;
pub mod coordinates;
pub mod deformation;
pub mod divergence;
pub mod elementwise;
pub mod gradient;
pub mod mask;
pub mod merge;
pub mod restrict;
pub mod select;

pub use band::Band;
pub use beam_deformation::beam_deformation;
pub use consolidate::consolidate;
pub use consolidate_element::consolidate_element;
pub use coordinates::{coordinates, displace, set_coordinates};
pub use deformation::deformation;
pub use divergence::divergence;
pub use elementwise::{abs, cos, cosh, exp, log, log10, sin, sinh, sqrt, tan, tanh};
pub use gradient::gradient;
pub use mask::{mask_cells, mask_nodes, mask_sub_cells, mask_sub_nodes};
pub use merge::merge;
pub use restrict::restrict;
pub use select::{select_cells, select_nodes, select_sub_cells, select_sub_nodes};
