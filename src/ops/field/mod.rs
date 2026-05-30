//! Field operators — derivations and transformations on
//! [`crate::containers::node_field::NodeField`] / [`crate::containers::element_field::ElementField`].
//!
//! Examples to come: `gradient`, `divergence`, `interp_to_gauss`,
//! `project_to_nodes`, `restrict`. Component-wise scalar maths and
//! per-element operations stay close to the data (on the field
//! types) — this module is for operations that **cross containers**
//! (mesh + field, fe_space + field).

pub mod coordinates;

pub use coordinates::coordinates;
