//! Field operators — derivations and transformations on
//! [`crate::containers::node_field::SubNodeField`] / [`crate::containers::element_field::ElementField`].
//!
//! [`gradient`] (`∇f` of a nodal field) and [`deformation`] (the linearized
//! strain `ε` of a displacement field) are the purely geometric producers of
//! the per-element field that [`crate::ops::behavior::integrate`] consumes.
//! More to come: `divergence`, `interp_to_gauss`, `project_to_nodes`.
//! Component-wise scalar maths and per-element operations stay close to the
//! data (on the field types) — this module is for operations that **cross
//! containers** (mesh + field, fe_space + field).

pub mod beam_deformation;
pub mod consolidate;
pub mod coordinates;
pub mod deformation;
pub mod divergence;
pub mod gradient;
pub mod merge;
pub mod restrict;

pub use beam_deformation::beam_deformation;
pub use consolidate::consolidate;
pub use coordinates::{coordinates, displace, set_coordinates};
pub use deformation::deformation;
pub use divergence::divergence;
pub use gradient::gradient;
pub use merge::merge;
pub use restrict::restrict;
