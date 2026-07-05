//! Geometric measure operators.
//!
//! Bounding boxes, centroids, areas/volumes, Jacobian helpers, face
//! normals, element-quality metrics — anything that takes a `Mesh` /
//! `SubMesh` (and possibly a `SubNodeField` of coordinates) and returns
//! a scalar or a derived geometric quantity.

mod locate;
mod project;

pub use locate::{locate_points, Location};
pub use project::{project_points, Projection};
