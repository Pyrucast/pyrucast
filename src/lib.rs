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
pub mod atoms;
pub mod containers;
pub mod coords;
pub mod dump;
pub mod error;
pub mod interrupt;
pub mod models;
pub mod ops;
pub mod parallel;
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
/// `extension-module`). Named `_pyrucast` (private): the public `pyrucast`
/// package (see `python/pyrucast/__init__.py`) re-exports it with
/// `from ._pyrucast import *` and adds the pure-Python high-level layer.
#[cfg(feature = "python-api")]
#[pymodule]
fn _pyrucast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(set_swap_dir, m)?)?;
    m.add_function(wrap_pyfunction!(swap_dir, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::solver::solve, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::solver::solve_eliminate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::solver::solve_unilateral, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::from_live_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::line, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::circle, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::arc, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::sweep, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::transfinite, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::sweep_solid, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::extrude, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::translate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::rotate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::to_quadratic, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::convert, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::triangulate_volume, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::triangulate_surface, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::pave_surface, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::pave_volume, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::to_poi1, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::barycenter, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::border, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::skin, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::orient, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::invert, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::merge_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::elements_on, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_in_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_on_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_on_plane, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_below_plane, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_on_line, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_in_cylinder, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_on_cylinder, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_in_cone, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_on_cone, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_in_torus, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::points_on_torus, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::read_gmsh, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::read_gmsh_str, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::export::export_vtk, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::consolidate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::consolidate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::element_field::consolidate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::coords::set, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::coords::displace, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::restrict, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::restrict_like, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::select, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::mask, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::element_field::mask, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::merge, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::element_field::gradient, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::element_field::deformation, m)?)?;
    m.add_function(wrap_pyfunction!(
        py::ops::element_field::interp_to_gauss,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py::ops::element_field::thermal_strain, m)?)?;
    m.add_function(wrap_pyfunction!(
        py::ops::element_field::beam_deformation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py::ops::element_field::frame_deformation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::divergence, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::measure::integral, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::measure::xty, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::measure::xtx, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::field::psca, m)?)?;
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
    m.add_function(wrap_pyfunction!(py::ops::matrix::stiffness, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::matrix::mass, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::matrix::lump, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::matrix::geometric, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::matrix::tangent, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::flux, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::internal_forces, m)?)?;
    m.add_function(wrap_pyfunction!(
        py::ops::node_field::internal_forces_continuum,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py::ops::element_field::integrate_behavior,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py::ops::element_field::sub_material_field,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py::ops::element_field::material_field, m)?)?;
    m.add_function(wrap_pyfunction!(
        py::ops::element_field::material_field_per_sub_model,
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
