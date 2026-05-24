//! Data containers — the typed objects that pyrucast manipulates.
//!
//! Each module here defines the Rust data structures and inherent
//! algorithms for one logical container. The pattern is generally
//! `Sub<Xxx>` (one piece, stored in the global store) + `Xxx`
//! (an [`crate::aggregate::Aggregate`] of those sub-pieces). Standalone
//! data types like [`matrix::Matrix`] and [`node_field::NodeField`] live
//! here too.
//!
//! Python wrappers for everything in this tree live under `crate::py`
//! (only compiled with the `python-api` feature); operators that take
//! containers in and out live under [`crate::ops`] (themed by build /
//! geom / field / assemble).

pub mod element_field;
pub mod fe_space;
pub mod matrix;
pub mod mesh;
pub mod model;
pub mod node_field;
