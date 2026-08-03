//! Containers — the **divisible** objects of pyrucast.
//!
//! A container is a type of which any part is still of its own nature:
//! half a mesh is a mesh, half a field is a field. That is what makes it
//! an *operand* — and, by the API convention, the only thing an operator
//! may take as its subject. The indivisible types it is built from live
//! in [`crate::atoms`]; the coordinate store they all hang from lives in
//! [`crate::coords`].
//!
//! Each module here defines the data structure and the inherent algorithms
//! of one container. The pattern is uniformly `Sub<Xxx>` (one piece,
//! stored in the global store) + `Xxx` (an
//! [`crate::aggregate::Aggregate`] of those pieces) — the aggregate is
//! *how* a container decomposes.
//!
//! Python wrappers for everything in this tree live under `crate::py`
//! (only compiled with the `python-api` feature); the operators that
//! consume and produce containers live under [`crate::ops`].

pub mod element_field;
pub mod evolution;
pub mod field;
pub mod finite_element_space;
pub mod matrix;
pub mod mesh;
pub mod model;
pub mod node_field;
