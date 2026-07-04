//! Linear solvers for assembled `Matrix` / `SubNodeField` systems.
//!
//! Exposes a sparse LU back-end ([`lu`], Lagrange path) and a master/slave
//! elimination solver ([`eliminate`], condensation path); future
//! direct-sparse and iterative solvers will live alongside them under
//! their own sub-modules.

pub mod eliminate;
pub mod lu;
