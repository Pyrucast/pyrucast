//! Python wrappers for the operations in [`crate::ops`].
//!
//! Free functions (factories and transforms) live here, one module per
//! `src/ops/<family>/` subtree, mirroring the Rust layout so the wrapper of
//! an operation sits at the matching path. Type wrappers stay in
//! `src/py/<type>.rs` (mirroring `src/containers/`); this `ops` tree holds
//! the *verbs*, those the *nouns*.

pub mod assemble;
pub mod behavior;
pub mod build;
pub mod export;
pub mod field;
pub mod mesher;
pub mod solver;
