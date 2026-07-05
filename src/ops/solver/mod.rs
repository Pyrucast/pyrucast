//! Linear solvers for assembled `Matrix` / `SubNodeField` systems.
//!
//! Exposes a sparse LU back-end ([`lu`], Lagrange path), a master/slave
//! elimination solver ([`eliminate`], condensation path) and an active-set
//! solver for unilateral constraints ([`unilateral`], status method); future
//! direct-sparse and iterative solvers will live alongside them under
//! their own sub-modules.

pub mod eliminate;
pub mod lu;
pub mod unilateral;
