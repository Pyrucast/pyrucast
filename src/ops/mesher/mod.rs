//! Mesh-creation operators (the "mesher").
//!
//! Each generator lives in its own sub-module and is re-exported at this
//! level so callers can write `crate::ops::mesher::line(...)`.
//!
//! Internal helpers:
//! - [`triangulation`] — 2-D primitives (ear clipping, CDT,
//!   polygon-with-holes pipeline).
//! - [`tetrahedralization`] — 3-D primitives (exact predicates, Delaunay
//!   kernel) behind the tetrahedral mesher.
//! - [`sweep_kernel`] — extrusion and SEG2→QUA4 kernel used by
//!   [`sweep()`](fn@sweep) and [`extrude()`].

pub mod arc;
pub mod barycenter;
pub mod border;
pub mod circle;
pub mod consolidate;
pub mod contour;
pub mod convert;
pub mod elements_on;
pub mod extrude;
pub mod from_live_nodes;
pub mod gmsh;
pub mod line;
pub mod merge_nodes;
pub mod orient;
pub mod predicates;
pub mod quadratic;
pub mod skin;
pub mod sweep;
pub mod sweep_kernel;
pub mod sweep_solid;
pub mod tetrahedralization;
pub mod to_poi1;
pub mod transfinite;
pub mod transform;
pub mod triangulate_surface;
pub mod triangulate_volume;
pub mod triangulation;

pub use arc::arc;
pub use barycenter::barycenter;
pub use border::border;
pub use circle::circle;
pub use consolidate::consolidate;
pub use convert::convert;
pub use elements_on::elements_on;
pub use extrude::extrude;
pub use from_live_nodes::from_live_nodes;
pub use gmsh::{read_gmsh, read_gmsh_str};
pub use line::line;
pub use merge_nodes::merge_nodes;
pub use orient::{invert, orient};
pub use quadratic::to_quadratic;
pub use skin::skin;
pub use sweep::sweep;
pub use sweep_solid::sweep_solid;
pub use to_poi1::to_poi1;
pub use transfinite::transfinite;
pub use transform::{rotate, translate};
pub use triangulate_surface::{triangulate_surface, triangulate_surface_cancellable};
pub use triangulate_volume::{triangulate_volume, triangulate_volume_cancellable};
