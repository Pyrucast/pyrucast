//! PyO3 wrappers — Python bindings for every public container of pyrucast.
//!
//! Each Rust container (`mesh`, `finite_element_space`, `model`, ...) has its mirror
//! file here defining the `Py*` classes and `#[pymethods]` impls. The
//! aim is to keep the container modules focused on Rust data + algorithms
//! and concentrate the FFI surface in one place.

pub mod cell;
pub mod configuration;
pub mod element;
pub mod element_field;
pub mod finite_element_space;
pub mod matrix;
pub mod mesh;
pub mod model;
pub mod node;
pub mod node_field;
pub mod ops;
