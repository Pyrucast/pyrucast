//! Assembly operators — turn a [`crate::model::Model`] into a
//! [`crate::matrix::Matrix`] (stiffness, mass) or
//! [`crate::node_field::NodeField`] (RHS).
//!
//! The per-physics integrands live in [`crate::models`]
//! (`heat_conduction`, `dirichlet`, …). This layer orchestrates the
//! loop over sub-models, the DOF layout, and boundary-condition
//! application.
//!
//! Today the orchestration still lives on `Model::stiffness` and
//! `Model::mass`. As we add `Model::rhs(...)`, multi-physics couplings
//! and BC variants, the heavy lifting will migrate here so
//! `containers/model.rs` stays a thin aggregate.
