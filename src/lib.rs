//! pyrucast — finite-element library: Rust core, Python API.
//!
//! Inspired by cast3m principles. The API conventions live in
//! `CONVENTIONS.md` (repository root) and in the mdbook.
//!
//! # Conventions
//!
//! - Every error goes through [`PyrucastError`] / [`Result`].
//! - Every serializable object uses the [`archive::portable::Portable`] trait
//!   (the byte contract of file save/load, identical on Linux and Windows).
//! - Display comes in three layered levels, all wired to Python:
//!   `Display`/`__str__` (one-line summary), `Debug`/`__repr__` (bounded
//!   structure — never bulk content), and [`dump::Dump`]/`dump(…)` (full
//!   content: matrix grids, value tables, connectivity). See [`dump`].
//!
//! # Example
//!
//! ```
//! use pyrucast::archive::Portable;
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
pub mod archive;
pub mod atoms;
pub mod containers;
pub mod coords;
pub mod dump;
pub mod error;
pub mod handle;
pub mod interrupt;
pub mod models;
pub mod named;
pub mod ops;
pub mod parallel;

#[cfg(feature = "python-api")]
pub mod py;

#[cfg(feature = "viz")]
pub mod viz;

pub use error::{PyrucastError, Result};

/// Library version (taken from `Cargo.toml`).
///
/// ```
/// // Celle de `Cargo.toml`, et celle qu'une archive inscrit en tête.
/// assert!(pyrucast::VERSION.split('.').count() >= 2);
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Features réellement compilées dans ce binaire, dans un ordre stable.
///
/// La sdist publiée sur PyPI ne compile que `extension-module` : une installation
/// depuis les sources n'a donc **pas** la visualisation, alors qu'elle porte le
/// même numéro de version que les wheels. Cette constante rend la différence
/// lisible sur la machine où elle se produit, plutôt que dans une page de
/// documentation.
///
/// ```
/// // `viz-interactive` implique `viz` — la liste ne peut pas dire le contraire.
/// let f = pyrucast::FEATURES;
/// assert!(!f.contains(&"viz-interactive") || f.contains(&"viz"));
/// ```
pub const FEATURES: &[&str] = &[
    #[cfg(feature = "python-api")]
    "python-api",
    #[cfg(feature = "extension-module")]
    "extension-module",
    #[cfg(feature = "viz")]
    "viz",
    #[cfg(feature = "viz-interactive")]
    "viz-interactive",
    #[cfg(feature = "abi3")]
    "abi3",
];

#[cfg(feature = "python-api")]
use pyo3::prelude::*;

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
    // Un tuple, non une liste : ce que le binaire porte ne se modifie pas
    // depuis Python.
    m.add("__features__", pyo3::types::PyTuple::new(m.py(), FEATURES)?)?;
    m.add_function(wrap_pyfunction!(py::archive::save, m)?)?;
    m.add_function(wrap_pyfunction!(py::archive::load, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::solver::solve, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::solver::solve_eliminate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::solver::solve_unilateral, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::model::heat_conduction, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::model::fick, m)?)?;
    m.add_function(wrap_pyfunction!(
        models::radiation::radiation_py::radiation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py::ops::model::interface_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(
        models::boundary_transfer::boundary_transfer_py::boundary_transfer,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(models::truss::truss_py::truss, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::model::elasticity, m)?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::von_mises::plasticity_perfect_py::plasticity_perfect,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::von_mises::plasticity_isotropic_py::plasticity_isotropic,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::drucker_prager::drucker_prager_py::drucker_prager,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::ottosen::ottosen_py::ottosen,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::viscous::creep_norton_py::creep_norton,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::viscous::creep_blackburn_py::creep_blackburn,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::viscous::creep_lemaitre_py::creep_lemaitre,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::viscous::viscoplasticity_chaboche_py::viscoplasticity_chaboche,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::viscous::viscoplasticity_lemaitre_chaboche_py::viscoplasticity_lemaitre_chaboche,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::damage::damage_tc::damage_tc_py::damage_tc,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::damage::sic_sic::damage_sic_sic_py::damage_sic_sic,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::plasticity::gurson::gurson_py::gurson,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::damage::mazars::mazars_py::mazars,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        models::bernoulli::bernoulli_py::bernoulli,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(models::shell::shell_py::shell, m)?)?;
    m.add_function(wrap_pyfunction!(
        models::timoshenko::timoshenko_py::timoshenko,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py::ops::model::dirichlet, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::model::mpc, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::model::embedded, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::model::contact, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::from_live_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::line, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::circle, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::arc, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::sweep, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::transfinite, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::sweep_solid, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::extrude, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::revolve, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::translate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::rotate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::symmetry_point, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::symmetry_line, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::symmetry_plane, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::to_quadratic, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::convert, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::copy, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::triangulate_volume, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::triangulate_surface, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::cleanup, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::grid_surface, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::grid_surface2, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::merge_triangles, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::regularize, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::pave_surface, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::pave_volume, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::poi1_from_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::to_poi1, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::barycenter, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::border, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::skin, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::orient, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::invert, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::chain, m)?)?;
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
    m.add_function(wrap_pyfunction!(py::ops::mesh::from_gmsh_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::read_gmsh, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::read_gmsh_str, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::export::export_vtk, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::mesh::consolidate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::consolidate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::element_field::consolidate, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::positions, m)?)?;
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
        py::ops::element_field::shell_deformation,
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
    m.add_function(wrap_pyfunction!(models::flux::flux_py::flux, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::internal_forces, m)?)?;
    m.add_function(wrap_pyfunction!(py::ops::node_field::external_forces, m)?)?;
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
