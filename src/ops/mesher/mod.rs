//! Mesh-creation operators (the "mesher").
//!
//! Two sub-domains so far:
//! - [`triangulation`] — 2-D primitives (ear clipping, CDT,
//!   polygon-with-holes pipeline) used to fill bounded regions.
//! - [`sweep`] — extrusion and SEG2→QUA4 sweep between two contours.
//!
//! [`crate::mesh::Mesh`]'s static constructors that build a mesh from
//! existing data (e.g. `Mesh::extrude`, `Mesh::sweep_qua4`) delegate
//! here so their bodies live with the rest of the mesher logic.

pub mod sweep;
pub mod triangulation;
