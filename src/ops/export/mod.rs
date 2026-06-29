//! Export operators — write meshes and fields to external file formats for
//! third-party viewers.
//!
//! - [`vtk`] — legacy VTK (`UNSTRUCTURED_GRID`, ASCII) for ParaView.

pub mod vtk;

pub use vtk::{
    vtk_element_field_string, vtk_mesh_string, vtk_node_field_string, write_vtk_element_field,
    write_vtk_mesh, write_vtk_node_field,
};
