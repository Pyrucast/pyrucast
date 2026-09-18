//! Geometry primitives shared by mesh containers and mesh operators.
//!
//! Type aliases over [`nalgebra`] so the rest of the crate can use
//! short names. Free vectors live next to points since they share the
//! same geometric role.

/// 2-D point, `nalgebra::Point2<f64>`.
///
/// ```
/// # use pyrucast::atoms::{Point2, Vector2};
/// // A **point**: a position. Its difference with another is a vector, which
/// // the typing imposes rather than leave it to the reader.
/// let a = Point2::new(1.0, 2.0);
/// let v: Vector2 = Point2::new(4.0, 6.0) - a;
/// assert_eq!(v, Vector2::new(3.0, 4.0));
/// assert_eq!(v.norm(), 5.0);
/// ```
pub type Point2 = nalgebra::Point2<f64>;
/// 3-D point, `nalgebra::Point3<f64>`.
///
/// ```
/// # use pyrucast::atoms::{Point3, Vector3};
/// let a = Point3::new(0.0, 0.0, 0.0);
/// let v: Vector3 = Point3::new(1.0, 2.0, 2.0) - a;
/// assert_eq!(v.norm(), 3.0);
/// ```
pub type Point3 = nalgebra::Point3<f64>;
/// 2-D vector, `nalgebra::Vector2<f64>`.
///
/// ```
/// # use pyrucast::atoms::Vector2;
/// // A **displacement**, not a position: it adds up, has a norm and
/// // se produit scalairement.
/// let u = Vector2::new(3.0, 4.0);
/// assert_eq!(u.norm(), 5.0);
/// assert_eq!(u.dot(&Vector2::new(1.0, 0.0)), 3.0);
/// ```
pub type Vector2 = nalgebra::Vector2<f64>;
/// 3-D vector, `nalgebra::Vector3<f64>`.
///
/// ```
/// # use pyrucast::atoms::Vector3;
/// // En 3-D s'y ajoute le produit vectoriel, dont vit toute normale.
/// let n = Vector3::x().cross(&Vector3::y());
/// assert_eq!(n, Vector3::z());
/// ```
pub type Vector3 = nalgebra::Vector3<f64>;
