//! Internal machinery of the tetrahedral mesher `mesh_volume`.
//!
//! The operator itself reads the envelope, validates it and materializes the
//! result; everything between those two ends lives here, on plain index and
//! coordinate arrays with no knowledge of the container layer.
//!
//! - [`predicates`] — exact `orient3d` / `insphere`, the foundation every
//!   other decision rests on.
//! - [`fill`] — exhaustive rebuild of a small region from its own vertices,
//!   the complete answer where a flip pattern is only a guess.
//! - [`surface`] — re-cutting a flat strip of the outer surface, for the
//!   envelope edges that run inside a flat face of the solid.
//! - [`recovery`] — putting the envelope's edges and facets back into the
//!   triangulation.
//! - [`classify`] — flooding from both sides of the recovered surface to
//!   separate material from void.
//! - [`flips`] — the 2-3 / 3-2 reconnections used to walk the envelope back
//!   into the triangulation.
//! - [`envelope`] — reading and validating the closed input surface.
//! - [`delaunay`] — the incremental Delaunay kernel and its tetrahedron
//!   adjacency structure.
//! - [`intersect`] — exact triangle-triangle intersection, used to reject a
//!   self-intersecting envelope before it can confuse the kernel.

pub mod classify;
pub mod delaunay;
pub mod envelope;
pub mod fill;
pub mod flips;
pub mod intersect;
pub mod predicates;
pub mod recovery;
pub mod surface;
