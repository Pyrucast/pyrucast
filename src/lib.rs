//! pyrucast — finite-element library: Rust core, Python API.
//!
//! Inspired by cast3m principles. The phased roadmap and conventions live
//! in `ROADMAP.md` (repository root) and in the mdbook.
//!
//! # Conventions
//!
//! - Every error goes through [`PyrucastError`] / [`Result`].
//! - Every serializable object uses the [`persist::Persist`] trait
//!   (shared backbone of disk swap and file save/load, portable between
//!   Linux and Windows).
//! - Every object implements `Debug` (structure) and `Display`
//!   (cast3m-style summary); both are wired to Python's `__repr__` and
//!   `__str__` respectively.
//!
//! # Example
//!
//! ```
//! use pyrucast::persist::Persist;
//!
//! #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
//! struct Demo {
//!     values: Vec<f64>,
//! }
//!
//! let a = Demo { values: vec![1.0, 2.0, 3.0] };
//! let bytes = a.to_bytes().unwrap();
//! let b = Demo::from_bytes(&bytes).unwrap();
//! assert_eq!(a, b);
//! ```

pub mod aggregate;
pub mod containers;
pub mod error;
pub mod models;
pub mod ops;
pub mod persist;
pub mod store;

#[cfg(feature = "python-api")]
pub mod py;

#[cfg(feature = "viz")]
pub mod viz;

pub use error::{PyrucastError, Result};

/// Library version (taken from `Cargo.toml`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "python-api")]
use pyo3::prelude::*;

/// Set the swap directory (Python binding of [`store::set_swap_dir`]).
#[cfg(feature = "python-api")]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
fn set_swap_dir(path: std::path::PathBuf) {
    store::set_swap_dir(path);
}

/// Return the effective swap directory (Python binding of [`store::swap_dir`]).
#[cfg(feature = "python-api")]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
fn swap_dir() -> PyResult<std::path::PathBuf> {
    Ok(store::swap_dir()?)
}

// Stub-info gatherer used by the `stub_gen` binary to produce
// `pyrucast.pyi`. Only present when the `stub-gen` feature is on.
#[cfg(feature = "stub-gen")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);

/// Python extension module. Built only under maturin (feature
/// `extension-module`).
#[cfg(feature = "python-api")]
#[pymodule]
fn pyrucast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(set_swap_dir, m)?)?;
    m.add_function(wrap_pyfunction!(swap_dir, m)?)?;
    m.add_function(wrap_pyfunction!(py::solver::solve, m)?)?;
    m.add_function(wrap_pyfunction!(py::mesh::from_live_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(py::mesh::line_seg2, m)?)?;
    m.add_function(wrap_pyfunction!(py::mesh::circle_seg2, m)?)?;
    m.add_function(wrap_pyfunction!(py::mesh::sweep_qua4, m)?)?;
    m.add_function(wrap_pyfunction!(py::mesh::extrude, m)?)?;
    m.add_function(wrap_pyfunction!(py::mesh::fill_surface, m)?)?;
    m.add_function(wrap_pyfunction!(py::mesh::to_poi1, m)?)?;
    m.add_function(wrap_pyfunction!(py::node_field::coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(py::node_field::restrict, m)?)?;
    m.add_function(wrap_pyfunction!(py::node_field::merge, m)?)?;
    m.add_function(wrap_pyfunction!(py::model::mass, m)?)?;
    m.add_class::<py::configuration::PyConfiguration>()?;
    m.add_class::<py::node::PyNode>()?;
    m.add_class::<py::mesh::PySubMesh>()?;
    m.add_class::<py::mesh::PyMesh>()?;
    m.add_class::<py::cell::PyCell>()?;
    m.add_class::<py::element::PyElement>()?;
    m.add_class::<py::node_field::PyNodeField>()?;
    m.add_class::<py::finite_element_space::PySubFiniteElementSpace>()?;
    m.add_class::<py::finite_element_space::PyFiniteElementSpace>()?;
    m.add_class::<py::element_field::PySubElementField>()?;
    m.add_class::<py::element_field::PyElementField>()?;
    m.add_class::<py::matrix::PySubMatrix>()?;
    m.add_class::<py::matrix::PyMatrix>()?;
    m.add_class::<py::model::PySubModel>()?;
    m.add_class::<py::model::PyModel>()?;
    Ok(())
}
