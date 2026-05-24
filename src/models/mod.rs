//! Per-physics implementations of `SubModel` variants.
//!
//! Each file here owns the **specifics** of one [`crate::model::Physics`]
//! variant: construction logic that goes beyond wrapping inputs, plus
//! local assembly of stiffness / mass contributions into the global
//! [`crate::matrix::Matrix`]. The [`crate::model::SubModel`] type
//! dispatches into these modules so that each physics's code lives in
//! one self-contained place.
//!
//! Adding a new physics means: extend [`crate::model::Physics`] with a
//! variant, add `models/<physics_name>.rs` with its `build` + assembly
//! functions, and wire the dispatch in `SubModel`.

pub mod dirichlet;
pub mod heat_conduction;
