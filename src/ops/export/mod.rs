//! Export operators — write meshes and fields to external formats.
//!
//! - [`arrays`] — flat arrays, the one exit every exchange format goes
//!   through (the mirror of [`crate::ops::mesh::arrays`]).
//! - [`vtk`] — legacy VTK (`UNSTRUCTURED_GRID`, ASCII or binary, single
//!   files or time series) for ParaView.

pub mod arrays;
pub mod vtk;

pub use arrays::{to_arrays, ElementLayout, Exported};
pub use vtk::{
    vtk_element_field_string, vtk_mesh_string, vtk_node_field_string, write_vtk_element_field,
    write_vtk_mesh, write_vtk_node_field, write_vtk_series, VtkEncoding,
};
