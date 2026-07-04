//! Mesh-creation operators (the "mesher").
//!
//! Each generator lives in its own sub-module and is re-exported at this
//! level so callers can write `crate::ops::mesher::line_seg2(...)`.
//!
//! Internal helpers:
//! - [`triangulation`] — 2-D primitives (ear clipping, CDT,
//!   polygon-with-holes pipeline).
//! - [`sweep`] — extrusion and SEG2→QUA4 kernel used by [`sweep_qua4()`](fn@sweep_qua4)
//!   and [`extrude()`].

pub mod barycenter;
pub mod circle_seg2;
pub mod consolidate;
pub mod contour;
pub mod elements_on;
pub mod extrude;
pub mod fill_surface;
pub mod from_live_nodes;
pub mod gmsh;
pub mod line_seg2;
pub mod merge_nodes;
pub mod quadratic;
pub mod surface;
pub mod sweep;
pub mod sweep_qua4;
pub mod sweep_solid;
pub mod to_poi1;
pub mod transform;
pub mod triangulation;
pub mod volume;

pub use barycenter::barycenter;
pub use circle_seg2::circle_seg2;
pub use consolidate::consolidate;
pub use contour::contour;
pub use elements_on::elements_on;
pub use extrude::extrude;
pub use fill_surface::fill_surface;
pub use from_live_nodes::from_live_nodes;
pub use gmsh::{read_gmsh, read_gmsh_str};
pub use line_seg2::line_seg2;
pub use merge_nodes::merge_nodes;
pub use quadratic::to_quadratic;
pub use surface::{surface, surface_cancellable};
pub use sweep_qua4::sweep_qua4;
pub use sweep_solid::sweep_solid;
pub use to_poi1::to_poi1;
pub use transform::{rotate, translate};
pub use volume::{volume, volume_cancellable};
