//! Build operators — construction of containers from data or other
//! containers.
//!
//! Will host (as legacy code migrates):
//! - `mesh::fill_surface`, `mesh::extrude`, `mesh::sweep_qua4`,
//!   `mesh::line_seg2`, `mesh::circle_seg2`, …
//! - any future remeshing / refinement helpers.

pub mod material_field;

pub use material_field::{material_field, material_field_per_sub_model, sub_material_field};
