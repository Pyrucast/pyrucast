//! Mesh-creation operators (the "mesher").
//!
//! Each generator lives in its own sub-module and is re-exported at this
//! level so callers can write `crate::ops::mesher::line_seg2(...)`.
//!
//! Internal helpers:
//! - [`triangulation`] — 2-D primitives (ear clipping, CDT,
//!   polygon-with-holes pipeline).
//! - [`sweep`] — extrusion and SEG2→QUA4 kernel used by [`sweep_qua4`]
//!   and [`extrude`].

pub mod circle_seg2;
pub mod consolidate;
pub mod extrude;
pub mod fill_surface;
pub mod from_live_nodes;
pub mod line_seg2;
pub mod sweep;
pub mod sweep_qua4;
pub mod to_poi1;
pub mod triangulation;

pub use circle_seg2::circle_seg2;
pub use consolidate::consolidate;
pub use extrude::extrude;
pub use fill_surface::fill_surface;
pub use from_live_nodes::from_live_nodes;
pub use line_seg2::line_seg2;
pub use sweep_qua4::sweep_qua4;
pub use to_poi1::to_poi1;
