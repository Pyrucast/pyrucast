//! Linear solvers for assembled `Matrix` / `SubNodeField` systems.
//!
//! Currently exposes a single dense LU back-end ([`lu`]); future
//! direct-sparse and iterative solvers will live alongside it under
//! their own sub-modules.

pub mod lu;
