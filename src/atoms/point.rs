//! Geometry primitives shared by mesh containers and mesh operators.
//!
//! Type aliases over [`nalgebra`] so the rest of the crate can use
//! short names. Free vectors live next to points since they share the
//! same geometric role.

/// 2-D point, `nalgebra::Point2<f64>`.
pub type Point2 = nalgebra::Point2<f64>;
/// 3-D point, `nalgebra::Point3<f64>`.
pub type Point3 = nalgebra::Point3<f64>;
/// 2-D vector, `nalgebra::Vector2<f64>`.
pub type Vector2 = nalgebra::Vector2<f64>;
/// 3-D vector, `nalgebra::Vector3<f64>`.
pub type Vector3 = nalgebra::Vector3<f64>;
