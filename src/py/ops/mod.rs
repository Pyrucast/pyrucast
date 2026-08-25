//! Python wrappers for the operators of [`crate::ops`].
//!
//! One module per `src/ops/<module>/` subtree, mirroring the Rust layout so
//! the wrapper of an operation sits at the matching path. Type wrappers stay
//! in `src/py/<type>.rs` (mirroring `src/containers/`); this tree holds the
//! *verbs*, those the *nouns*.
//!
//! The extension module `_pyrucast` is **flat**, so two operators sharing a
//! short name across modules (the three `consolidate`) carry a distinct
//! `#[pyo3(name = "…")]`. The pure-Python layer (`python/pyrucast/*.py`)
//! re-exports them under their real, unqualified name in the right
//! sub-module.

pub mod coords;
pub mod element_field;
pub mod export;
pub mod field;
pub mod matrix;
pub mod measure;
pub mod mesh;
pub mod model;
pub mod node_field;
pub mod solver;
