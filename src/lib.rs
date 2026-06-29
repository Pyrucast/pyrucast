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
//! - Display comes in three layered levels, all wired to Python:
//!   `Display`/`__str__` (one-line summary), `Debug`/`__repr__` (bounded
//!   structure — never bulk content), and [`dump::Dump`]/`dump(…)` (full
//!   content: matrix grids, value tables, connectivity). See [`dump`].
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

// Numerical/FE code is full of legitimate index-based loops (symmetric matrix
// fills, triangular axis pairs, indices shared across several arrays) where the
// iterator rewrite clippy suggests is less clear or outright impossible.
#![allow(clippy::needless_range_loop)]

pub mod aggregate;
pub mod containers;
pub mod dump;
pub mod error;
pub mod interrupt;
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

/// Set the directory used to swap large objects to disk. If never set, a
/// per-process subdirectory of the system temp dir is used.
#[cfg(feature = "python-api")]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
fn set_swap_dir(path: std::path::PathBuf) {
    store::set_swap_dir(path);
}

/// Return the effective swap directory (creating it if necessary).
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
    m.add_function(wrap_pyfunction!(py::ops::solver::solve, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::from_live_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::poi1_from_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::line_seg2, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::circle_seg2, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::sweep_qua4, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::extrude, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::fill_surface, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::surface, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::volume, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::to_poi1, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::barycenter, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::contour, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::merge_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::elements_on, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::read_gmsh, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesher::read_gmsh_str, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::export::export_vtk, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::consolidate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::set_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::displace, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::restrict, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::select, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::merge, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::gradient, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::deformation, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::beam_deformation, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::divergence, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::abs, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::exp, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::log, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::log10, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::cos, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::sin, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::tan, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::sinh, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::cosh, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::assemble::stiffness, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::assemble::mass, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::assemble::flux, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::behavior::integrate_behavior, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::build::sub_material_field, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::build::material_field, m)?)?;
    m.add_function(wrap_pyfunction!(
        py::ops::build::material_field_per_sub_model,
        m
    )?)?;
    m.add_class::<py::coords::PyCoords>()?;
    m.add_class::<py::node::PyNode>()?;
    m.add_class::<py::mesh::PySubMesh>()?;
    m.add_class::<py::mesh::PyMesh>()?;
    m.add_class::<py::cell::PyCell>()?;
    m.add_class::<py::element::PyElement>()?;
    m.add_class::<py::node_field::PySubNodeField>()?;
    m.add_class::<py::node_field::PyNodeField>()?;
    m.add_class::<py::finite_element_space::PySubFiniteElementSpace>()?;
    m.add_class::<py::finite_element_space::PyFiniteElementSpace>()?;
    m.add_class::<py::element_field::PySubElementField>()?;
    m.add_class::<py::element_field::PyElementField>()?;
    m.add_class::<py::matrix::PySubMatrix>()?;
    m.add_class::<py::matrix::PyMatrix>()?;
    m.add_class::<py::model::PySubModel>()?;
    m.add_class::<py::model::PyModel>()?;
    m.add_class::<py::evolution::PySubEvolution>()?;
    m.add_class::<py::evolution::PyEvolution>()?;
    Ok(())
}
