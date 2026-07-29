//! Internal machinery of the tetrahedral mesher `mesh_volume`.
//!
//! The operator itself reads the envelope, validates it and materializes the
//! result; everything between those two ends lives here, on plain index and
//! coordinate arrays with no knowledge of the container layer.
//!
//! - [`predicates`] — exact `orient3d` / `insphere`, the foundation every
//!   other decision rests on.
//! - [`flips`] — the 2-3 / 3-2 reconnections used to walk the envelope back
//!   into the triangulation.
//! - [`envelope`] — reading and validating the closed input surface.
//! - [`delaunay`] — the incremental Delaunay kernel and its tetrahedron
//!   adjacency structure.
//! - [`intersect`] — exact triangle-triangle intersection, used to reject a
//!   self-intersecting envelope before it can confuse the kernel.

pub mod delaunay;
pub mod envelope;
pub mod flips;
pub mod intersect;
pub mod predicates;
