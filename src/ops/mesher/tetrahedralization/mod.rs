//! Internal machinery of the tetrahedral mesher `mesh_volume`.
//!
//! The operator itself reads the envelope, validates it and materializes the
//! result; everything between those two ends lives here, on plain index and
//! coordinate arrays with no knowledge of the container layer.
//!
//! - [`predicates`] — exact `orient3d` / `insphere`, the foundation every
//!   other decision rests on.
//! - [`envelope`] — reading and validating the closed input surface.
//! - [`intersect`] — exact triangle-triangle intersection, used to reject a
//!   self-intersecting envelope before it can confuse the kernel.

pub mod envelope;
pub mod intersect;
pub mod predicates;
